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

echo "Deleted all existing data."

# generate meshes

radius=0.5
height=1
freq_max=2000

numPolyEachType_train=1
numPolyEachType_val=1
numPolyEachType_test=1

lowNumSide=4
upNumSide=4

for group in "train" "validation" "test"
do
    if [ ${group} == "train" ]; then
        #echo "generating mesh for train"
        ./pipeline_poly --each_type_no="${numPolyEachType_train}" --edge_lower="${lowNumSide}"\
        --edge_upper="${upNumSide}" --radius=${radius} --height=${height} --freq_max=${freq_max} --output_path="${base}/${group}/mesh/"
        sleep 1s
    fi
    if [ ${group} == "validation" ]; then
        #echo "generating mesh for val"
        ./pipeline_poly --each_type_no="${numPolyEachType_val}" --edge_lower="${lowNumSide}"\
        --edge_upper="${upNumSide}" --radius=${radius} --height=${height} --freq_max=${freq_max} --output_path="${base}/${group}/mesh/"
        sleep 1s
    fi
    if [ ${group} == "test" ]; then
        #echo "generating mesh for test"
        ./pipeline_poly --each_type_no="${numPolyEachType_test}" --edge_lower="${lowNumSide}"\
        --edge_upper="${upNumSide}" --radius=${radius} --height=${height} --freq_max=${freq_max} --output_path="${base}/${group}/mesh/"
        sleep 1s
    fi
done

echo "Generated meshes"
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
                curr_idx=$(((num_side-lowNumSide)*numEachType+idx))
                echo "dealing with mesh: ${base}/${group}/mesh/poly_${curr_idx}.obj"
                ./main --obj_file="${base}/${group}/mesh/poly_${curr_idx}.obj" --src_type="point" --src_radi=5 --src_num=8 --oct_num=4\
                --src_mag=1.0 --x_cnr=-2.56 --y_cnr=-2.56 --x_len=5.12 --y_len=5.12 --z_coord=0.5 --side_len=0.01\
                --vox_file="${base}/${group}/input/raw/vox${curr_idx}" --field_file="${base}/${group}/output/raw/field${curr_idx}"
                echo "output binary occupancy grid to ${base}/${group}/input/raw/vox${curr_idx}"
                echo "output field to ${base}/${group}/output/raw/field${curr_idx}"
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
                curr_idx=$(((num_side-lowNumSide)*numEachType+idx))
                echo "dealing with mesh: ${base}/${group}/mesh/poly_${curr_idx}.obj"
                ./main --obj_file="${base}/${group}/mesh/poly_${curr_idx}.obj" --src_type="point" --src_radi=5 --src_num=8 --oct_num=4\
                --src_mag=1.0 --x_cnr=-2.56 --y_cnr=-2.56 --x_len=5.12 --y_len=5.12 --z_coord=0.5 --side_len=0.01\
                --vox_file="${base}/${group}/input/raw/vox${curr_idx}" --field_file="${base}/${group}/output/raw/field${curr_idx}"
                echo "output binary occupancy grid to ${base}/${group}/input/raw/vox${curr_idx}"
                echo "output field to ${base}/${group}/output/raw/field${curr_idx}"
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
                curr_idx=$(((num_side-lowNumSide)*numEachType+idx))
                echo "dealing with mesh: ${base}/${group}/mesh/poly_${curr_idx}.obj"
                ./main --obj_file="${base}/${group}/mesh/poly_${curr_idx}.obj" --src_type="point" --src_radi=5 --src_num=8 --oct_num=4\
                --src_mag=1.0 --x_cnr=-2.56 --y_cnr=-2.56 --x_len=5.12 --y_len=5.12 --z_coord=0.5 --side_len=0.01\
                --vox_file="${base}/${group}/input/raw/vox${curr_idx}" --field_file="${base}/${group}/output/raw/field${curr_idx}"
                echo "output binary occupancy grid to ${base}/${group}/input/raw/vox${curr_idx}"
                echo "output field to ${base}/${group}/output/raw/field${curr_idx}"
            done
        done
    fi
done
			

