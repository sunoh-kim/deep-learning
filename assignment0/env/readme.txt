포함된 파일: 3개
env/environment.yml : conda install libraries
env/requirements.txt : pip install libraries 
env/setup_env.sh : installation bash script file

cd env
(Anaconda envs directory path(ANACONDA_ENV_PATH) check in setup_env.sh)
bash setup_env.sh
source activate deep-learning-19 (conda activate deep-learning-19)
pip install –r requirements.txt
source deactivate (conda deactivate)
