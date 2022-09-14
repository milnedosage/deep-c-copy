#!/bin/sh

SSH_KEY_FILE_DIR='pub_keys'

echo "Are you sure you want to grant SSH access? Press any key to grant access, Ctrl+C to terminate"
read
echo "You sure? Press any key to continue, Ctrl+C to terminate"
read

cd "$SSH_KEY_FILE_DIR"

for FILENAME in $(ls)
do
    cat "$FILENAME" >> ~/.ssh/authorized_keys
done
