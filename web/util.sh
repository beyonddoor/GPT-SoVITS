ssh_proxy() {
    ports=(9874 9873 9872)
    name="${1:-autodl-bj-clone}"

    for port in "${ports[@]}"; do
        nohup ssh -L "0.0.0.0:${port}:localhost:${port}" "$name" -N -f &
        echo "Port forwarding set up: 0.0.0.0:${port} -> localhost:${port} on ${name}"
    done

    ps aux | grep ssh
}

list_files() {
    find "$1" -type f -exec ls -l {} \;
}

rsync_code() {
    rsync -avc --ignore-times ./ "$1":GPT-SoVITS/ --include '**/' --include '**/*.py' --exclude '*' --prune-empty-dirs 
}

clear_data() {
    rm -rf TEMP/"$1"
    rm -rf logs/GPT-SoVITS/"$1"
    rm -rf output/"$1"
}

# ssh_proxy "$1"