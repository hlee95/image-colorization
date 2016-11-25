#!/bin/bash -e
function getPlaces2 {
    echo "Making image data directory"
    mkdir -p /home/ubuntu/image-colorization/data

    cd /home/ubuntu/image-colorization/data

    echo "Downloading image dataset"
    curl "http://6.869.csail.mit.edu/fa16/challenge/data.tar.gz" | tar xz
}

function flattenDirectories {
    echo "Flattening directories and preserving labels"
    cd /home/ubuntu/image-colorization/data
    local count=0
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
                    cd ..
                else
                    local name=$count"_"$file
                    mv $file ../../$name
                fi
            done

            echo $category"="$count >> ../../../label_mapping.txt
            (( count=count+1 ))
            cd ..
        done
        cd ..
        rm -r $letter
    done
    echo "Done. Label mapping in data/images/label_mapping.txt"
}

# Only download data if the caller passed -d or --download.
download_data=0
while [[ $# -gt 0 ]]
do
    flag="$1"
    case $flag in
        -d | --download)
            download_data=1
            ;;
        *)
            echo Usage: $0 "[-d | --download]"
            exit 1
            ;;
    esac
    shift
done
if [ $download_data -eq 1 ]; then
    getPlaces2
fi
flattenDirectories

