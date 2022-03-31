# 1C-OpenCV-search-gradient-img
Анализ объектов на картинке, в частности поиск прямоугольных градиентных участков.

## Алгоритмы:
 - Поиска нулевой подматрицы наибольшего размера
> Цель: Найти подматрицу быстрее, чем за O(n*m^2).
> 
> Идея: Будем расширять область, ограниченную единицами.
> 
> Реализация: Первым делом выставляем дефолтные значения на границы up, left, right.
> 
> Далее будем перебирать строки и для каждого столбца определять границы по бокам и сверху.
> Для подчета маски сверху воспользуемся генератором, который выставляет ограничение.
```python
up = [i if matrix[i][j] == 1 else up[j] for j in range(m)]
```
> Вычисляем левую границу:
```python
st = []
for j in range(m):
    st = find_barrier(st)
    left[j] = -1 if not st else st[-1]
    st.append(j)
```
> Аналогично вычислем правую:
```python
st = []
for j in range(m - 1, -1, -1):
    st = find_barrier(st)
    right[j] = m if not st else st[-1]
    st.append(j)
```
> Посчитываем площадь, обновляем ответ, если выгодно:
```python
for j in range(m):
    new_area = (i - up[j]) * (right[j] - left[j] - 1)
    if area < new_area:
        area = new_area
        coordinates_y = [up[j] + 1, i]
        coordinates_x = [left[j] + 1, right[j] - 1]
```
> Т.к. мы перебирали строки и столбцы, ассимптотика не вышла за О(n*m).
 - Алгоритмы фильтрации, привязанные к расчету производных по цвету пикселя.
   - filter Laplace: https://www.sciencedirect.com/topics/engineering/laplacian-filter
   - filter Sobel: https://fiveko.com/sobel-filter/
   - filter Scharr: https://plantcv.readthedocs.io/en/v3.0.5/scharr_filter/

## Запуск
```
./python gradient_tracker.py
```

## Настройка
В config.json, вид которого такой:
```json
{
  "cfg_grad":
    {
      "filter" : "laplace",
      "color_density" : 5
    }
}
```
 - Сделайте выбор в пользу Вашего любимого алгоритма фильтрации ('laplace', 'sobel', 'scharr').
 - Поберите процент (color_density), при котором точность фильтрации будет наибольшей. Процент - целое число от 0 до 100.
