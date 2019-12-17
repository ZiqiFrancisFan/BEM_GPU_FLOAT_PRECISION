#!/bin/bash

# set up the base path
base="$(pwd)/data"
echo "The base path: ${base}"

# clear existing data
for group in "train" "validation" "test"
do
    for subgroup in "input" "output" "mesh"
    do
        # curr represents the current path
        if [ ${subgroup} == "mesh" ]; then
            curr="${base}/${group}/${subgroup}"
            rm -rfv "${curr}/"*
        else
            for subsubgroup in "raw" "mat"
            do
                curr="${base}/${group}/${subgroup}/${subsubgroup}"
                #echo "Current path: ${curr}"
                rm -rfv "${curr}/"*
            done   
        fi
    done
done

echo "Deleted all previous data."

# generate meshes

radius=0.5
height=1
freq_max=2000

numPolyEachType_train=1;
numPolyEachType_val=1;
numPolyEachType_test=1

lowNumSide=3
upNumSide=3

for group in "train" "validation" "test"
do
    if [ ${group} == "train" ]; then
        #echo "generating mesh for train"
        numEachType=numPolyEachType_train
        for (( num_side=lowNumSide; num_side<=upNumSide; num_side++ ))
        do
            for (( idx=0; idx<numEachType; idx++ ))
            do
                curr_idx=$(((num_side-3)*numEachType+idx))
                ./pipeline_poly --edge_no="$num_side" --radius="${radius}" --height="${height}" --freq_max="${freq_max}"\
                    --output_path="${base}/${group}/mesh/poly${curr_idx}.obj"
            done
        done
    fi
    if [ ${group} == "validation" ]; then
        #echo "generating mesh for val"
        numEachType=numPolyEachType_val
        for (( num_side=lowNumSide; num_side<=upNumSide; num_side++ ))
        do
            for (( idx=0; idx<numEachType; idx++ ))
            do
                #echo "idx: ${idx}"
                curr_idx=$(((num_side-3)*numEachType+idx))
                #echo "current index: $curr_idx"
                ./pipeline_poly --edge_no="$num_side" --radius="${radius}" --height="${height}" --freq_max="${freq_max}"\
                    --output_path="${base}/${group}/mesh/poly${curr_idx}.obj"
            done
        done
    fi
    if [ ${group} == "test" ]; then
        #echo "generating mesh for test"
        numEachType=numPolyEachType_test
        for (( num_side=lowNumSide; num_side<=upNumSide; num_side++ ))
        do
            for (( idx=0; idx<numEachType; idx++ ))
            do
                #echo "idx: ${idx}"
                curr_idx=$(((num_side-3)*numEachType+idx))
                ./pipeline_poly --edge_no="$num_side" --radius="${radius}" --height="${height}" --freq_max="${freq_max}"\
                    --output_path="${base}/${group}/mesh/poly${curr_idx}.obj"
            done
        done
    fi
done

# generate loudness fields
#read -p "Press [Enter] key to start backup..."
for group in "train" "validation" "test"
do
    if [ ${group} == "train" ]; then
        #echo "generating mesh for train"
        numEachType=numPolyEachType_train
        for (( num_side=lowNumSide; num_side<=upNumSide; num_side++ ))
        do
            for (( idx=0; idx<numEachType; idx++ ))
            do
                echo "mesh path: ${base}/${group}/mesh/poly${curr_idx}.obj"
                echo "vox path: ${base}/${group}/input/raw/vox${curr_idx}"
                echo "field path: ${base}/${group}/output/raw/field${curr_idx}"
                curr_idx=$(((num_side-3)*numEachType+idx))
                ./main --obj_file="${base}/${group}/mesh/poly${curr_idx}.obj" --src_type="point" --src_radi=5 --src_num=8 --oct_num=4\
                --src_mag=1.0 --x_cnr=-2.56 --y_cnr=-2.56 --x_len=5.12 --y_len=5.12 --z_coord=0.5 --side_len=0.01\
                --vox_file="${base}/${group}/input/raw/vox${curr_idx}" --field_file="${base}/${group}/output/raw/field${curr_idx}"
                #read -p "Press [Enter] key to start backup..."
            done
        done
    fi
    if [ ${group} == "validation" ]; then
        #echo "generating mesh for val"
        numEachType=numPolyEachType_val
        for (( num_side=lowNumSide; num_side<=upNumSide; num_side++ ))
        do
            for (( idx=0; idx<numEachType; idx++ ))
            do
                echo "mesh path: ${base}/${group}/mesh/poly${curr_idx}.obj"
                echo "vox path: ${base}/${group}/input/raw/vox${curr_idx}"
                echo "field path: ${base}/${group}/output/raw/field${curr_idx}"
                #echo "idx: ${idx}"
                curr_idx=$(((num_side-3)*numEachType+idx))
                ./main --obj_file="${base}/${group}/mesh/poly${curr_idx}.obj" --src_type="point" --src_radi=5 --src_num=8 --oct_num=4\
                --src_mag=1.0 --x_cnr=-2.56 --y_cnr=-2.56 --x_len=5.12 --y_len=5.12 --z_coord=0.5 --side_len=0.01\
                --vox_file="${base}/${group}/input/raw/vox${curr_idx}" --field_file="${base}/${group}/output/raw/field${curr_idx}"
                #read -p "Press [Enter] key to start backup..."
            done
        done
    fi
    if [ ${group} == "test" ]; then
        #echo "generating mesh for test"
        numEachType=numPolyEachType_test
        for (( num_side=lowNumSide; num_side<=upNumSide; num_side++ ))
        do
            for (( idx=0; idx<numEachType; idx++ ))
            do
                echo "mesh path: ${base}/${group}/mesh/poly${curr_idx}.obj"
                echo "vox path: ${base}/${group}/input/raw/vox${curr_idx}"
                echo "field path: ${base}/${group}/output/raw/field${curr_idx}"
                #echo "idx: ${idx}"
                curr_idx=$(((num_side-3)*numEachType+idx))
                ./main --obj_file="${base}/${group}/mesh/poly${curr_idx}.obj" --src_type="point" --src_radi=5 --src_num=8 --oct_num=4\
                --src_mag=1.0 --x_cnr=-2.56 --y_cnr=-2.56 --x_len=5.12 --y_len=5.12 --z_coord=0.5 --side_len=0.01\
                --vox_file="${base}/${group}/input/raw/vox${curr_idx}" --field_file="${base}/${group}/output/raw/field${curr_idx}"
                #read -p "Press [Enter] key to start backup..."
            done
        done
    fi
done
			

