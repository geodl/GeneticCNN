# Convenience script to run tensorboard
# Run from terminal because of messed up path in virtual env of anaconda
import os

if __name__ == "__main__":
    os.system('tensorboard --logdir=./logs/')
