#!/bin/bash

for i in {1..20}
do
    curl -X POST --form "file=@./curriculos/curriculo_${i}.pdf" --form "type=public" --header "username: admin" http://localhost:5000/upload_pdf
done

curl -G http://localhost:5000/search --header "username: candidate" --data-urlencode "query=Quais desenvolvedores exigiriam uma  maior pretensão salarial?"

