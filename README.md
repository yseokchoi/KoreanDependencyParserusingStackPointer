
# Korean Dependency Parser using Stack-Pointer Networks
Stack-Pointer Network를 이용한 한국어 의존 구문 파서 

이 코드는 Stack-Pointer Network 코드를 

[1]: https://github.com/XuezheMax/NeuroNLP2	"NeuroNLP2"

 코드에서 한국어 의존 구문 파서에 맞게 변경되었음.

### Requirement
- python 3.6
- gensim
- numpy
- Pytorch >= 0.4

### Train

------

python StackPointerParser.py  --pos --char --eojul --word_embedding *random* --char_embedding *random* --pos_embedding *random* --train *"train_path"* --dev *"dev_path"* --test *"test_path"* --model_path *"model_path"* --model_name *"model_name"* --grandPar --sibling --skipConnect

### Test

------

python3 StackPointerParser_test.py --model_path *"model_path"* --model_name *"model_name"* --output_path *"output_path"* --test *"test_file"* 

