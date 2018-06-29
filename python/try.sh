# J=(4,8,16,32,64,128,256,512)
J=(4 8 16 32 64 128 256 512)
for j in ${J[@]}
do
    echo $j
    mkdir $j
    cd $j
    for i in {1..1000}
    do
	# echo $i
	# python trying.py $i $j
	python ../trying.py $i $j >> datos.txt
    done
    python ../1.py
    cd ..
done

