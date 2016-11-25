#!/bin/bash
function getPlaces2 {
    echo "Making training data directory"
    mkdir -p /home/ubuntu/image-colorization/data

    cd /home/ubuntu/image-colorization/data

    echo "Downloading training images"
    wget "http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz"

    echo "Extracting images"
    tar -xzf data.tar.gz
}
function flattenDirectories {

    local count = 0
    cd images/train

    for letter in *
    do 
        cd $letter
        for label in *
        do
            local category=$label
            cd $label

            for file in *
            do
                if [ -d "$file" ]; then
                    cd $file
                    category=$category"_"$file

                    for sub in *
                    do
                        local name=$count"_"$sub
                        mv $sub ../../../$name
                    done

                    cd..
                else
                    local name=$count"_"$file
                    mv $file ../../$name
                fi
            done

            echo $category"="$count >> ../../../label_mapping.txt
            (( count=count+1 ))
            cd ..
            rm -r $label
        done

        cd ..
        rm -r $letter
    done
}
getPlaces2
flattenDirectories