#!/bin/bash

# commits a change
git add --all;

echo -e "please enter the commit message followed by [enter]: \n"

read msg

git commit -m "$msg"

echo "Hello "$USER".\nPlease enter the name of the upstream remote followed by [ENTER].\nType Enter to push to 'master' "

read remote

if [[ $remote == "" ]]; then
   remote=master
fi

git push -u origin $remote


