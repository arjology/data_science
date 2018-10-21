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

function distance {
    if [ -z "$LOG" ]; then 
        shortest_route -i $DATA -s $SRC -d $DST
    else
        shortest_route -i $DATA -s $SRC -d $DST -l $LOG
    fi
}

# Setup vars
DATA=$1
SRC=$2
DST=$3
LOG=$4

if (( $# < 3 )); then
   echo $#
   useage
   exit
fi

distance
