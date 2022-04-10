from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from miscc.config import cfg
from PIL import Image

from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from utils.data_utils import prepare_data

from utils.loss import words_loss
from utils.loss import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
from utils.model import G_DCGAN

#################################################
# DO NOT CHANGE 
from utils.model import RNN_ENCODER, CNN_ENCODER, GENERATOR, DISCRIMINATOR, DISCRIMINATOR_64, DISCRIMINATOR_128
#################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
                
class condGANTrainer(object):
#     def __init__(self, output_dir, train_dataset, train_dataloader, test_dataset, test_dataloader, dataloader_for_wrong_samples):
    def __init__(self, output_dir, data_loader, dataloader_val, dataloader_for_wrong_samples, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(int(cfg.GPU_ID))
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
#         self.output_dir = output_dir
#         self.train_dataloader = train_dataloader
        self.test_dataloader = dataloader_val
        self.dataloader_for_wrong_samples = dataloader_for_wrong_samples
        
#         self.batch_size = cfg.BATCH_SIZE
#         self.max_epoch = cfg.TRAIN.MAX_EPOCH
#         self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        
#         self.n_words = train_dataset.n_words # size of the dictionary
        '''
        '''

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(os.path.join(cfg.CHECKPOINT_DIR, img_encoder_path), map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load my best image encoder')
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.NET_E),
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load my best text encoder')
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import DISCRIMINATOR_64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import DISCRIMINATOR_128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import DISCRIMINATOR as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import DISCRIMINATOR_64, DISCRIMINATOR_128, DISCRIMINATOR
            netG = GENERATOR()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(DISCRIMINATOR_64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(DISCRIMINATOR_128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(DISCRIMINATOR())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        # if cfg.TRAIN.NET_G != '':
        #     state_dict = \
        #         torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        #     netG.load_state_dict(state_dict)
        #     print('Load G from: ', cfg.TRAIN.NET_G)
        #     istart = cfg.TRAIN.NET_G.rfind('_') + 1
        #     iend = cfg.TRAIN.NET_G.rfind('.')
        #     epoch = cfg.TRAIN.NET_G[istart:iend]
        #     epoch = int(epoch) + 1
        #     if cfg.TRAIN.B_NET_D:
        #         Gname = cfg.TRAIN.NET_G
        #         for i in range(len(netsD)):
        #             s_tmp = Gname[:Gname.rfind('/')]
        #             Dname = '%s/netD%d.pth' % (s_tmp, i)
        #             print('Load D from: ', Dname)
        #             state_dict = \
        #                 torch.load(Dname, map_location=lambda storage, loc: storage)
        #             netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]
    
    
    def prepare_data(self, data):
        imgs, captions, captions_lens, class_ids, keys, sentence_idx = data

        # sort data by the length in a decreasing order
        sorted_cap_lens, sorted_cap_indices = \
            torch.sort(captions_lens, 0, True)

        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).cuda())
            else:
                real_imgs.append(Variable(imgs[i]))

        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        # sent_indices = sent_indices[sorted_cap_indices]
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        # print('keys', type(keys), keys[-1])  # list
        if cfg.CUDA:
            captions = Variable(captions).cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

        return [real_imgs, captions, sorted_cap_lens,
                class_ids, keys, sentence_idx]

#     def prepare_data(self, data):
#         """
#         Prepares data given by dataloader
#         e.g., x = Variable(x).cuda()
#         """
#         imgs, captions, captions_lens, class_ids, keys, sentence_idx = data

#         # sort data by the length in a decreasing order
#         # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html 
#         sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
        
#         #################################################
#         # TODO
#         # this part can be different, depending on which algorithm is used
#         # imgs = imgs[sorted_cap_indices]
#         # if cfg.CUDA:
#         #     imgs = Variable(imgs).cuda()
#         #################################################

#         captions = captions[sorted_cap_indices].squeeze()
#         class_ids = class_ids[sorted_cap_indices].numpy()
#         keys = [keys[i] for i in sorted_cap_indices.numpy()]

#         if cfg.CUDA:
#             captions = Variable(captions).cuda()
#             sorted_cap_lens = Variable(sorted_cap_lens).cuda()
#         else:
#             captions = Variable(captions)
#             sorted_cap_lens = Variable(sorted_cap_lens)

#         return [imgs, captions, sorted_cap_lens, class_ids, keys, sentence_idx]
    
    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
    
    def train(self):
        """
        e.g., for epoch in range(cfg.TRAIN.MAX_EPOCH):
                  for step, data in enumerate(self.train_dataloader, 0):
                      x = self.prepare_data()
                      .....
        """
        #################################################
        # TODO: Implement text to image synthesis
        
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, _ = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)
        
        #################################################
        
    
    def generate_inception_score_data(self):
        # load the text encoder model to generate images for evaluation
        print(self.n_words, cfg.TEXT.EMBEDDING_DIM)
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.NET_E), map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        self.text_encoder.eval()
        
        # load the generator model to generate images for evaluation
        self.netG = GENERATOR()
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.NET_G), map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        print('Load generator from:', cfg.TRAIN.NET_G)
        self.netG.eval()

        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.netG = self.netG.cuda()
            noise = noise.cuda()

        for step, data in enumerate(self.test_dataloader, 0):
            imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)
            
            #################################################
            # TODO
            # word embedding might be returned as well 
            # hidden = self.text_encoder.init_hidden(self.batch_size)
            # sent_emb = self.text_encoder(captions, cap_lens, hidden)
            # sent_emb = sent_emb.detach()
            #################################################

            hidden = self.text_encoder.init_hidden(self.batch_size)
            words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
#             num_words = words_embs.size(2)
#             if mask.size(1) > num_words:
#                 mask = mask[:, :num_words]

            noise.data.normal_(0, 1)
            
            #################################################
            # TODO
            # this part can be different, depending on which algorithm is used
            # the main purpose is generating synthetic images using caption embedding and latent vector (noise)
            # fake_img = self.netG(noise, sent_emb, ...)
            #################################################
            
            fake_imgs, _, _, _ = self.netG(noise, sent_emb, words_embs, mask)
            
            fake_imgs = fake_imgs[-1]
            
            for j in range(self.batch_size):
                if not os.path.exists(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0])):
                    os.mkdir(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0]))
                
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
    
    
    def generate_r_precision_data(self):
        # load the image encoder model to obtain the latent feature of the generated image
        
        self.image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(os.path.join(cfg.CHECKPOINT_DIR, img_encoder_path), map_location=lambda storage, loc: storage)
        
#         state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER), map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        self.image_encoder.eval()
        
        # load the image encoder model to obtain the latent feature of the real caption
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.NET_E), map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        self.text_encoder.eval()
        
        # load the image encoder model to generate synthetic images based on the text (caption) feature
        self.netG = GENERATOR()
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.NET_G), map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        self.netG.eval()

        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.image_encoder = self.image_encoder.cuda()
            self.netG = self.netG.cuda()
            noise = noise.cuda()
        
        # cfg.NUM_BATCH_FOR_TEST will be different, depending on batch size (29330//batch_size)
        cfg.NUM_BATCH_FOR_TEST = len(self.test_dataloader)
        self.num_batches = cfg.NUM_BATCH_FOR_TEST
        
        true_cnn_features = np.zeros((self.num_batches, self.batch_size, cfg.TEXT.EMBEDDING_DIM), dtype=float)
        true_rnn_features = np.zeros((self.num_batches, self.batch_size, cfg.TEXT.EMBEDDING_DIM), dtype=float)
        wrong_rnn_features = np.zeros((self.num_batches, cfg.WRONG_CAPTION, self.batch_size, cfg.TEXT.EMBEDDING_DIM), dtype=float)

        dataiter = iter(self.dataloader_for_wrong_samples)
        
        for step, data in enumerate(self.test_dataloader, 0):
            imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)
            
            #################################################
            # TODO
            # word embedding might be returned as well 
            # hidden = self.text_encoder.init_hidden(self.batch_size)
            # sent_emb = self.text_encoder(captions, cap_lens, hidden)
            # sent_emb = sent_emb.detach()
            #################################################
            
            hidden = self.text_encoder.init_hidden(self.batch_size)
            words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
#             num_words = words_embs.size(2)
#             if mask.size(1) > num_words:
#                 mask = mask[:, :num_words]
            
            noise.data.normal_(0, 1)
            
            #################################################
            # TODO
            # this part can be different, depending on which algorithm is used
            # the main purpose is generating synthetic images using caption embedding and latent vector (noise)
            # fake_img = self.netG(noise, sent_emb, ...)
            #################################################
            
            fake_imgs, _, _, _ = self.netG(noise, sent_emb, words_embs, mask)

            #################################################
            # TODO
            # this part is getting image features of the generated image
            # # of returns can be different, depending on how you code the image encoder method
            # image_embs = self.image_encoder(fake_imgs)
            # image_embs = image_embs.detach()
            #################################################

            _, image_embs = self.image_encoder(fake_imgs)
            image_embs = image_embs.detach()
            
            
            true_cnn_features[step] = image_embs.cpu()
            true_rnn_features[step] = sent_emb.cpu()
            
            for each_wrong_idx in range(cfg.WRONG_CAPTION):
                data_for_wrong_samples = next(dataiter, None)
                if data_for_wrong_samples is None: # after one epoch, dataloader next iter should be reset
                    dataiter = iter(self.dataloader_for_wrong_samples)
                    data_for_wrong_samples = dataiter.next()
                _, captions_for_wrong_samples, cap_lens_for_wrong_samples, class_ids_for_wrong_samples, keys_for_wrong_samples, sent_idx_for_wrong_samples = self.prepare_data(data_for_wrong_samples)
                
                #################################################
                # TODO
                # word embedding might be returned as well
                # hidden = self.text_encoder.init_hidden(self.batch_size)
                # wrong_sent_emb = self.text_encoder(captions_for_wrong_samples, cap_lens_for_wrong_samples, hidden)
                # wrong_sent_emb = wrong_sent_emb.detach()
                #################################################
            
                hidden = self.text_encoder.init_hidden(self.batch_size)
                wrong_words_embs, wrong_sent_emb = self.text_encoder(captions_for_wrong_samples, cap_lens_for_wrong_samples, hidden)
                wrong_words_embs, wrong_sent_emb = wrong_words_embs.detach(), wrong_sent_emb.detach()
            
                wrong_rnn_features[step, each_wrong_idx] = wrong_sent_emb.cpu()

        try:
            os.remove(os.path.join(cfg.R_PRECISION_DIR, cfg.R_PRECISION_FILE))
        except OSError:
            pass
        np.savez(os.path.join(cfg.R_PRECISION_DIR, cfg.R_PRECISION_FILE), true_cnn=true_cnn_features,
                 true_rnn=true_rnn_features,
                 wrong_rnn=wrong_rnn_features)
        
#     def save_model(self):
#         """
#         Saves models
#         """
#         torch.save(self.netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR))
#         torch.save(self.text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER))
#         torch.save(self.image_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER))

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = GENERATOR()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys, _ = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = GENERATOR()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)