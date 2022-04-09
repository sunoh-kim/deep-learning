files="Assignment5_Part1_DQN.ipynb
Assignment5_Part2_Playing_Atari_game.ipynb"
"a3c.py"
"envs.py"
"model.py"
"worker.py"

echo "Check files...."
for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done
echo "Must check the trained model (./cartpole/trained_agent, ./pong/trained_agent)"

echo "Make zip file...."
if [ -z "$1" ]; then
    echo "Team number is required.
Usage: ./CollectSubmission team_#"
    exit 0
fi

rm -f $1.tar.gz
mkdir $1
cp -r cartpole/ pong/ *.ipynb a3c.py envs.py model.py worker.py  $1/
tar cvzf $1.tar.gz $1

echo "Done."
