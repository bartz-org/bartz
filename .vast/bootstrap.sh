# set environment variables
export EDITOR=vim
env >> /etc/environment

# configure tmux
cat >> ~/.tmux.conf << 'EOF'
set-option -g history-limit 10000
EOF

# configure the shell
cat >> ~/.bashrc << 'EOF'
conda deactivate
cd /workspace/bartz
EOF

# install R
apt update -qq
apt install -y --no-install-recommends software-properties-common dirmngr
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt install -y --no-install-recommends r-base r-base-dev

# install R and python envs
make setup
