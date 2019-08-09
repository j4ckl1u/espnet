#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# system path
. ./path.sh
. ./cmd.sh

# chmod for exec file
. ./exe.sh

set -x

# soft link to kaldi
[ -e steps ] && rm -rf steps
[ -e utils ] && rm -rf utils
ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
ln -s $KALDI_ROOT/egs/wsj/s5/utils utils

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=vggblstm     # encoder architecture type
elayers=5
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=2
dunits=1024
# attention related
atype=location
adim=1024
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=24
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=10

# rnnlm related
# lm_layers=2
# lm_units=650
lm_layers=1
lm_units=1024
lm_opt=sgd        # or adam
# lm_batchsize=256  # batch size in LM training
# lm_epochs=60      # if the data size is large, we can reduce this
# lm_maxlen=100     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_batchsize=1024  # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_maxlen=40     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
# lm_weight=0.5
lm_weight=0.7
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.5
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
dataDir=.
expDir=.

# Base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# data dir
downdir=${dataDir}/downloads
datadir=${dataDir}/data
dumpdir=${dataDir}/dump   # directory to dump full features
fbankdir=${dataDir}/fbank


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"


if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${downdir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${downdir}/LibriSpeech/${part} ${datadir}/$(echo ${part} | sed s/-/_/g)
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ]; then
    ###Task dependent. You have to design training and dev sets by yourself.
    ###But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        if [ -f ${fbankdir}/.${x}.complete ]; then
            echo "feature $x was already successfully generated, nothing to do."
            continue
        fi
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            ${datadir}/${x} ${expDir}/make_fbank/${x} ${fbankdir}
        # touch ${fbankdir}/.${x}.complete
    done

    if [ ! -f ${datadir}/${train_set}_org/.complete ]; then
        utils/combine_data.sh ${datadir}/${train_set}_org ${datadir}/train_clean_100 ${datadir}/train_clean_360 ${datadir}/train_other_500
        # touch ${datadir}/${train_set}_org/.complete
    else
        echo "${train_set}_org was already successfully combined, nothing to do."
    fi
    if [ ! -f ${datadir}/${train_dev}_org/.complete ]; then
        utils/combine_data.sh ${datadir}/${train_dev}_org ${datadir}/dev_clean ${datadir}/dev_other
        # touch ${datadir}/${train_dev}_org/.complete
    else
        echo "${train_dev}_org was already successfully combined, nothing to do."
    fi

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${datadir}/${train_set}_org ${datadir}/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${datadir}/${train_dev}_org ${datadir}/${train_dev}

    # compute global CMVN
    if [ ! -f ${datadir}/${train_set}/cmvn.ark ]; then
        compute-cmvn-stats scp:${datadir}/${train_set}/feats.scp ${datadir}/${train_set}/cmvn.ark
    else
        echo "Find the previous computed cmvn.ark file in ${train_set}, remove it if you want to recompute!"
    fi

    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        ${datadir}/${train_set}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expDir}/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        ${datadir}/${train_dev}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expDir}/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            ${datadir}/${rtask}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expDir}/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=${datadir}/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${datadir}/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${datadir}/lang_char/
    if [ ! -f ${dict} ]; then
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        cut -f 2- -d" " ${datadir}/${train_set}/text > ${datadir}/lang_char/input.txt
        spm_train --input=${datadir}/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
        spm_encode --model=${bpemodel}.model --output_format=piece < ${datadir}/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    else
        echo "dict and bpe model has been generated previously, if you want to change it, modify to retrain!"
    fi
    wc -l ${dict}

    # make json labels
    if [ ! -f ${feat_tr_dir}/data_${bpemode}${nbpe}.json ]; then
        data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
         ${datadir}/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
        echo "make json file ${feat_tr_dir}/data_${bpemode}${nbpe}.json done!"
    else
        echo "Json file ${feat_tr_dir}/data_${bpemode}${nbpe}.json exist, you can modify to regenerate it!"
    fi

    if [ ! -f ${feat_dt_dir}/data_${bpemode}${nbpe}.json ]; then
        data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
             ${datadir}/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
        echo "Make json file ${feat_dt_dir}/data_${bpemode}${nbpe}.json done!"
    else
        echo "Json file ${feat_dt_dir}/data_${bpemode}${nbpe}.json exist, you can modify to regenerate it!"
    fi

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        if [ ! -f ${feat_recog_dir}/data_${bpemode}${nbpe}.json ]; then
            data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
                ${datadir}/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
            echo "Make json file ${feat_recog_dir}/data_${bpemode}${nbpe}.json done!"
        else
            echo "Json file ${feat_recog_dir}/data_${bpemode}${nbpe}.json exist, you can modify to regenerate it!"
        fi
    done
fi



# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
fi
lmexpdir=${expDir}/train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=${lmexpdir}/lm_train_${bpemode}${nbpe}
    mkdir -p ${lmdatadir}
    # use external data
    if [ ! -e ${lmdatadir}/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${lmdatadir}/
    fi
    cut -f 2- -d" " ${datadir}/${train_set}/text | gzip -c > ${lmdatadir}/${train_set}_text.gz
    # combine external text and transcriptions and shuffle them with seed 777
    zcat ${lmdatadir}/librispeech-lm-norm.txt.gz ${lmdatadir}/${train_set}_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " ${datadir}/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
                                                    > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. single gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=${expDir}/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=${expDir}/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${dumpdir}/dev_clean/delta${do_delta}/data_${bpemode}${nbpe}.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
	    --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --epochs ${epochs}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=24

    for rtask in ${recog_set}; do
    (
        echo "$rtask start!"
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        echo "$rtask split finished"

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            &
        wait

        echo "$rtask decode finished"

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

        echo "$rtask score_sclite finished"

    ) &
    done
    wait
    echo "Finished"
fi

