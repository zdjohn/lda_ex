#!/bin/sh

if test -d dist
then 
    echo "remove existing package"
    rm -rf dist
fi

if test -d dependencies
then 
    echo "remove existing package"
    rm -rf dependencies
fi

if test -f dist_files.zip
then 
    echo "remove previous dist_files.zip"
    rm dist_files.zip
fi 


# pip install -t dependencies -r requirements.txt
# cd dependencies; zip -r ../dist_files.zip .
# cd ..; zip -ru dist_files.zip src -x */__pycache__/\*
zip -r dist_files.zip clustering_ex -x */__pycache__/\*

# create deployable artifacts
mkdir dist
mv dist_files.zip dist/.
cp main.py dist/.
cp settings.json dist/.
cp requirements.txt dist/.
cp bootstrap.sh dist/.
