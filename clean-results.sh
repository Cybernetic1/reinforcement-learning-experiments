echo "**** use -N to show files less than N days old"
echo "**** use -N -delete to actually delete those files"
find results/* -daystart -mtime $1 $2
