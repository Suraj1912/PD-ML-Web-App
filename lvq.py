import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache(suppress_st_warning=True)
def LVQ(X_train, X_test, y_train, y_test, epoch, learn_rate):
    train_data = train_lvq(data= X_train, labels= y_train, num_epochs= int(epoch), learning_rate= learn_rate, validation_data= X_test, validation_labels= y_test)
    st.write(train_data)

@st.cache(suppress_st_warning=True)
def train_lvq(data, labels, num_epochs, learning_rate, validation_data=None, validation_labels=None):
    # Get unique class labels.
    num_dims = data.shape[1]
    labels = labels.astype(int)
    unique_labels = list(set(labels))

    num_protos = len(unique_labels)
    prototypes = np.empty((num_protos, num_dims))
    proto_labels = []

    # Initialize prototypes using class means.
    for i in unique_labels:
        class_data = data[labels == i, :]

        # Compute class mean.
        mean = np.mean(class_data, axis=0)

        prototypes[i] = mean
        proto_labels.append(i)

    # Loop through data set.
    for epoch in range(0, num_epochs):
        for fvec, lbl in zip(data, labels):
            # Compute distance from each prototype to this point
            distances = list(np.sum(np.subtract(fvec, p)**2) for p in prototypes)
            min_dist_index = distances.index(min(distances))

            # Determine winner prototype.
            winner = prototypes[min_dist_index]
            winner_label = proto_labels[min_dist_index]

            # Push or repel the prototype based on the label.
            if winner_label == lbl:
                sign = 1
            else:
                sign = -1

            # Update winner prototype
            prototypes[min_dist_index] = np.add(prototypes[min_dist_index], np.subtract(fvec, winner) * learning_rate * sign)

        # Use validation set to test performance.
        val_err = 0
        if validation_labels is not None:
            for fvec, lbl in zip(validation_data, validation_labels):
                distances = list(np.sum(np.subtract(fvec, p) ** 2) for p in prototypes)
                min_dist_index = distances.index(min(distances))

                # Determine winner prototype label
                winner_label = proto_labels[min_dist_index]

                # Check if labels match
                if not winner_label == lbl:
                    val_err = val_err + 1

            val_err = val_err / len(validation_labels)
            st.write("Epoch " + str(epoch) + ". Testing error: " + str(val_err))
        else:
            st.write("Epoch " + str(epoch))


    return (prototypes, proto_labels)