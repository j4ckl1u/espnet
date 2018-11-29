#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
verbose=0

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <data-dir-src> <data-dir-trg> <dict>";
    exit 1;
fi

dir_src=$1
dir_trg=$2
dic=$3
tmpdir_src=`mktemp -d ${dir_src}/tmp-XXXXX`
rm -f ${tmpdir_src}/*.scp
#input, which is not necessary for decoding mode, and make it as an option
if [ ! -z ${feat} ]; then
    if [ ${verbose} -eq 0 ]; then
        utils/data/get_utt2num_frames.sh ${dir_src} &> /dev/null
        cp ${dir_src}/utt2num_frames ${tmpdir_src}/ilen_src.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir_src}/idim_src.scp &> /dev/null
    else
        utils/data/get_utt2num_frames.sh ${dir_src}
        cp ${dir_src}/utt2num_frames ${tmpdir_src}/ilen_src.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir_src}/idim_src.scp
    fi
fi

# output
if [ ! -z ${bpecode} ]; then
    paste -d " " <(awk '{print $1}' ${dir_src}/text) <(cut -f 2- -d" " ${dir_src}/text | spm_encode --model=${bpecode} --output_format=piece) > ${tmpdir_src}/token_src.scp
elif [ ! -z ${nlsyms} ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dir_src}/text > ${tmpdir_src}/token_src.scp
else
    text2token.py -s 1 -n 1 ${dir_src}/text > ${tmpdir_src}/token_src.scp
fi
cat ${tmpdir_src}/token_src.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir_src}/tokenid_src.scp
cat ${tmpdir_src}/tokenid_src.scp | awk '{print $1 " " NF-1}' > ${tmpdir_src}/olen_src.scp 
# +2 comes from CTC blank and EOS
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 2" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${dir_src}/text > ${tmpdir_src}/odim_src.scp

# others
if [ ! -z ${lang} ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir_src}/text > ${tmpdir_src}/lang_src.scp
fi
# feats
cat ${feat} > ${tmpdir_src}/feat.scp

rm -f ${tmpdir_src}/*.json
cp ${dir_src}/text ${tmpdir_src}/text_src.scp
cp ${dir_src}/utt2spk ${tmpdir_src}/utt2spk_src.scp
for x in ${tmpdir_src}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir_src}/${k}.json
done


tmpdir_trg=`mktemp -d ${dir_trg}/tmp-XXXXX`
rm -f ${tmpdir_trg}/*.scp
if [ ! -z ${feat} ]; then
    if [ ${verbose} -eq 0 ]; then
        utils/data/get_utt2num_frames.sh ${dir_src} &> /dev/null
        cp ${dir_src}/utt2num_frames ${tmpdir_trg}/ilen_trg.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir_trg}/idim_trg.scp &> /dev/null
    else
        utils/data/get_utt2num_frames.sh ${dir_src}
        cp ${dir_src}/utt2num_frames ${tmpdir_trg}/ilen_trg.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir_trg}/idim_trg.scp
    fi
fi

# output
if [ ! -z ${bpecode} ]; then
    paste -d " " <(awk '{print $1}' ${dir_trg}/text) <(cut -f 2- -d" " ${dir_trg}/text | spm_encode --model=${bpecode} --output_format=piece) > ${tmpdir_trg}/token_trg.scp
elif [ ! -z ${nlsyms} ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dir_trg}/text > ${tmpdir_trg}/token_trg.scp
else
    text2token.py -s 1 -n 1 ${dir_trg}/text > ${tmpdir_trg}/token_trg.scp
fi
cat ${tmpdir_trg}/token_trg.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir_trg}/tokenid_trg.scp
cat ${tmpdir_trg}/tokenid_trg.scp | awk '{print $1 " " NF-1}' > ${tmpdir_trg}/olen_trg.scp 
# +2 comes from CTC blank and EOS
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 2" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${dir_trg}/text > ${tmpdir_trg}/odim_trg.scp

# others
if [ ! -z ${lang} ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir_trg}/text > ${tmpdir_trg}/lang_trg.scp
fi



# feats
cat ${feat} > ${tmpdir_trg}/feat.scp

rm -f ${tmpdir_trg}/*.json
cp ${dir_trg}/text ${tmpdir_trg}/text_trg.scp
cp ${dir_trg}/utt2spk ${tmpdir_trg}/utt2spk_trg.scp
for x in ${tmpdir_trg}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir_trg}/${k}.json
done



mergejson_ctc.py --verbose ${verbose} ${tmpdir_src}/*.json ${tmpdir_trg}/*.json

#rm -fr ${tmpdir_trg}
#rm -fr ${tmpdir_src}
