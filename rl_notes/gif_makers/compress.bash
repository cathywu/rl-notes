gifsicle -i $1 --optimize=3  --colors 256 -o compressed.gif
mv compressed.gif $1
