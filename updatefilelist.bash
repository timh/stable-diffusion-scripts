while true
do
    echo -n "."
    find . -type f -name \*.png > filelist.txt
    inotifywait -t 15 -q -e create,move,delete -r --include .\*png .
    sleep 1
done
