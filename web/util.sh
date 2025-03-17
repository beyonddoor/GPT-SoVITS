ssh_proxy() {
    ports=(9874 9873 9872)
    name="${1:-autodl-bj-clone}"

    for port in "${ports[@]}"; do
        nohup ssh -L "0.0.0.0:${port}:localhost:${port}" "$name" -N -f &
        echo "Port forwarding set up: 0.0.0.0:${port} -> localhost:${port} on ${name}"
    done

    ps aux | grep ssh
}

ssh_proxy_stop() {
    ps aux | grep ssh | grep 987 | awk '{print $2}'
    if [[ $(readline) -eq 'y' ]]; then
        ps aux | grep ssh | grep 987 | awk '{print $2}' | xargs kill -9
    fi
}

list_files() {
    find "$1" -type f -exec ls -l {} \;
}

rsync_code() {
    rsync -avc --ignore-times ./ "${1:-none}":GPT-SoVITS/ --include '**/' --include '**/*.py' --exclude '*' --prune-empty-dirs 
}

clear_data() {
    rm -rf TEMP/"$1"
    rm -rf logs/GPT-SoVITS/"$1"
    rm -rf output/"$1"
}

start_train_server() {
    nohup celery -A worker.celery_app worker --loglevel=info &
    nohup uvicorn main:app --reload &
    nohup redis-server &
}

infer() {
    prefix="${4}"
    mkdir -p output/infer/$1"$prefix"
    python GPT_SoVITS/inference_cli.py \
    --gpt_model "${2:-gptmodel}" --sovits_model "${3:-sovits-model}" \
    --ref_audio data/$1/infer/ref.m4a --ref_text data/$1/infer/ref.txt \
    --ref_language 中文 --target_text data/$1/infer/infer.txt --target_language 中文 --output_path output/infer/$1"$prefix"
}

# ssh_proxy "$1"