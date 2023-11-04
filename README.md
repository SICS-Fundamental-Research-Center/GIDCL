# Graph-Enhanced Interpretable Data Cleaning with Large Language Models

- Detector: detector.ipynb
- Error Detection data: datasets/beers/detector/multi-view/train_aug.csv
- Error Correction Data: 
- Function Set: Function_Set.ipynb
- Graph Model: PyTorch-BigGraph
### LLM Regex Query Example: 
```
The input 

[['May-51', '51-5'], ['Jun-70', '70-6'], ['Jun-93', '93-6'], ['11-Sep', '3111-9'], ['Aug-91', '91-8'], ['Jul-72', '72-7'], ['Jul-71', '71-7'], ['Apr-41', '3541-4']]

are [clean,dirty] cell pairs from table rayyan column article_pagination, and ['1187-9', '', '283-4', '714-9 ST  - [Noninvasive prenatal diagnosis of trisomy 21, 18 and 13 using cell-free fetal DNA]-', '835-40', '185-91', '163-7', '43-49', '1158-78', '158-61', '317-325', '10213-10224', '711-5', 'S40', '1245-50', '919-21', '185-90', 'Cd004797', '257-8', '991-5', '1304-16', '512-9 ST  - A performance improvement process to tackle tachysystole-', '1530-9', '2496-502', '785-794'] are examples of all cells from this columns. Please conclude a general pattern for dirty and clean cells, and write a general function with regular expression to detect whether a given cell is dirty or not. Input and output are all string format.
```
