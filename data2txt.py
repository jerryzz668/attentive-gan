import os


def data2txt(data_rootdir):
    images = os.listdir(data_rootdir + 'data/')
    labels = os.listdir(data_rootdir + 'gt/')
    images.sort()
    labels.sort()

    image_len = len(images)
    label_len = len(labels)

    assert image_len == label_len


    trainText = open(data_rootdir + 'train.txt', 'w')
    for i in range(image_len):
        image_dir = data_rootdir + 'data/' + images[i] + ' '
        label_dir = data_rootdir + 'gt/' + labels[i] + '\n'

        trainText.write(image_dir)
        trainText.write(label_dir)

    trainText.close()
    print('finished!')


if __name__ == '__main__':
    data2txt('./data/training_data/')