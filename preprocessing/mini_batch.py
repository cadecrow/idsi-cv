import numpy as np

def mini_batch_rows(num_of_training_batches, num_of_testing_batches, num_of_training_examples):
    num_of_batches = num_of_training_batches + num_of_testing_batches
    np.random.seed(1)

    list_of_rows = np.arange(1, num_of_training_examples+1, 1)
    np.random.shuffle(list_of_rows)
    batch_size = num_of_training_examples // num_of_batches
    remainder = num_of_training_examples % num_of_batches

    batches = []
    batch_start = 0
    for i in range(num_of_batches):
        if (remainder > 0):
            batches.append(list_of_rows[batch_start : (batch_start + batch_size + 1)])
            batch_start += batch_size + 1
            remainder -= 1
        else:
            batches.append(list_of_rows[batch_start : (batch_start + batch_size)])
            batch_start += batch_size
    return (batches[:num_of_training_batches], batches[num_of_training_batches:])
