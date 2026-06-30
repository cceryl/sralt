#!/bin/bash

BINDS="--bind $HOME/dev:$HOME/dev"

if [ -d "$HOME/.vscode" ]; then
    BINDS="$BINDS --bind $HOME/.vscode:$HOME/.vscode"
fi
if [ -d "$HOME/.vscode-server" ]; then
    BINDS="$BINDS --bind $HOME/.vscode-server:$HOME/.vscode-server"
fi

apptainer exec --nv --containall --cleanenv \
    --home $HOME/container/ubuntu-home:$HOME \
    --pwd "$PWD" \
    $BINDS \
    $HOME/container/ubuntu \
    bash -c 'source "$1/.venv/bin/activate" && shift && exec python "$@"' -- "$PWD" "$@"
