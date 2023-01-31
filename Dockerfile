# -------------------------
# INITIAL CONFIGURATION
# -------------------------

# Sets the base image from build the project.
FROM python:3.9-buster

# Changes the working directory.
WORKDIR /code

# Sets bash as the default terminal.
SHELL ["/bin/bash", "-c"]

# Installs vim in the container for development usage.
RUN apt-get update && \
    apt install -y vim sqlite

# -------------------------
# GIT CONFIGURATION
# -------------------------

# Imports build args.
ARG GIT_NAME \
    GIT_EMAIL

# Sets environment variables values.
ENV GIT_NAME=${GIT_NAME} \
    GIT_EMAIL=${GIT_EMAIL}

# Adds the source for install an updated version of git.
RUN echo deb http://deb.debian.org/debian buster-backports main > /etc/apt/sources.list.d/buster-backports.list

# Updates packages and installs git.
RUN apt update && apt install -y -t buster-backports git

# Sets the global git user name.
RUN git config --global user.name "${GIT_NAME}"

# Sets the global git user email.
RUN git config --global user.email "${GIT_EMAIL}"

# Sets configuration that ignores permissions in files.
RUN git config --global core.fileMode false

# Downloads git-prompt.sh file
RUN wget -O ~/git-prompt.sh https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh

# Downloads git-completion.sh file
RUN wget -O ~/git-completion.bash https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash

# Adds git-completion.sh file to startup.
RUN echo "source ~/git-completion.bash" >> ~/.bashrc

# Adds git-completion.sh file to startup.
RUN echo "source ~/git-prompt.sh" >> ~/.bashrc

# Exports variable that shows the git repo state.
RUN echo "export GIT_PS1_SHOWDIRTYSTATE=1" >> ~/.bashrc

# Exports a PS1 that shows the git repo state in the terminal.
RUN echo "export PS1='\\[\$(tput bold)\\]\\[\\033[32m\\]\\u@api:\\[\\033[34m\\]\\w\\[\\033[31m\\]\`__git_ps1\`\\[\$(tput sgr0)\\]\\[\\033[97m\\] $ '" >> ~/.bashrc

# -------------------------
# PROJECT CONFIGURATION
# -------------------------

# Copies the files needed to install dependencies
COPY ./Pipfile /code/.
COPY ./Pipfile.lock /code/.

# Updates pip
RUN pip install --upgrade pip

# Installs pipenv
RUN pip install pipenv

# Installs the project requirements
RUN pipenv install --system --dev
