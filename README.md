# CMPE-597-Spring-2023

## Training Neural Network
To train the neural network, you can run cli command with project folder path using --project_path argument

python main.py --project_path C:\Users\a.acar\PycharmProjects\cmpe_587_assignment1

Other optional cli arguments are:
--epoch 
--batch_size_train
--batch_size_valid
--learning_rate

Default Values:
epoch => 40
batch_size_train => 50
batch_size_valid => 50
learning_rate => 0.01

Example usage:

python main.py --project_path C:\Users\a.acar\PycharmProjects\cmpe_587_assignment1 --epoch 25 --batch_size_train 300 --batch_size_valid 200 --learning_rate 0.001

## Evaluation on Test Data
you can run this command:
python eval.py --project_path xyz/abc/cmpe_587_assignment1

## TSNE plot
Please run tsne.py file from command line with project path argument to see tsne-2d plot and save to project folder.

python tsne.py --project_path xyz/abc/cmpe_587_assignment1

## Results

For 40 epoch , 0.01 learning rate , 50 batch_size for train and validation data :

training accuracy is : 0.3511 validation accuracy is : 0.3459

Screenshots:

![Screenshot 2023-04-04 at 01 25 37](https://user-images.githubusercontent.com/29437625/229853409-83700245-4115-4f00-b091-7ef6e09ec14e.png)
![Screenshot 2023-04-04 at 01 51 42](https://user-images.githubusercontent.com/29437625/229853426-0de36958-f3bf-4bbf-90a5-1e219c1fcf0f.png)


