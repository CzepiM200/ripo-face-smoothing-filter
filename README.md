# RiPO projekt - wygładzanie twarzy real time

## Uruchomienie
```bash
$ ./main.py
```

## No więc tak
Zostało do zrobienia to co mieliśmy wypisane w dokumentacji w pliku z planem:
* predyktor z biblioteki dlib
* nałożenie filtru między odpowiednimi segmentami twarzy

[Tutaj link do tutoriala](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/?fbclid=IwAR0m8IV8X1DKoMx8Q3y2Rnpg_SWtMEtjJJ5_xSYNoMwNafi6Avz6-MBq3i0), który krok po kroku opowiada ci jak wytrenować sobie predyktor dlibowy w najprostszy możliwy sposób. Wymaga danych trenujących które ci wysyłałem, jakby co link jest w artykule. 

Twój wybór jak bardzo złożony będzie ten predyktor, myślę że do odmładzania twarzy chcemy wygładzanie przede wszystkim na policzki no i na czoło by się przydało. Znajoma robiła na wykrywanie 30 punktów i musiała zostawić komputer do trenowania na 40 minut, ale ma wydajnego laptopa do gier.

Po wykryciu tych punktów zostaje już tylko nałożyć filtr na obszar między wybranymi z nich co nie powinno być trudne.