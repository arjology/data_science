#!/bin/bash

# Useage prompt
function useage {
    echo
    echo "Usage: $0 <input_data> <source_node> <destination_node>"
    echo
    echo "Example: $0 input.dat 0 1"
    echo
    exit
}

function install {
    make test
    make develop
}


function distance {
    shortest_route -i $DATA -s $SRC -d $DST
}

# Setup vars
DATA=$1
SRC=$2
DST=$3

if [ $# -ne 3 ]; then
   echo $#
   useage
   exit
fi

install
distance
