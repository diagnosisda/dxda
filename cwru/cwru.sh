for K in 0 1 2 3 4 
do
for VARSRC in 0 1 2 3
do
    for VARTAR in 0 1 2 3
    do
        if [ ! $VARTAR == $VARSRC ] ; then
            CUDA_VISIBLE_DEVICES=0 python3 -u cwru.py -source $VARSRC -target $VARTAR 
        fi
    done
done
done

