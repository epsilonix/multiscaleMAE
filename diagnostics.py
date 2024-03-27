    #DIAGNOSTICS MMMMMMMMMMMMMMMMMMMM
    total_images = len(data_loader_train.dataset)
    print(f"Total number of images in the DataLoader: {total_images}")
    
    for images, _ in data_loader_train:
        for i, image in enumerate(images):
            print(f"Image {i} shape: {image.shape}")
    
    from collections import Counter

    # Initialize a Counter object to keep track of label counts
    label_counts = Counter()

    # Iterate through the DataLoader
    for _, labels in data_loader_train:
        # Update counts (this example assumes labels are directly iterable)
        label_counts.update(labels)

    # Print the count of each label
    for label, count in label_counts.items():
        print(f"Label: {label}, Count: {count}")
    
    
    print('data_loader_train')
    
    import matplotlib.pyplot as plt
    
    

    def imshow(img):
        # Assuming img is a torch.Tensor, select the first channel
        if img.ndim == 4:  # Check if the image batch is in (B, C, H, W) format
            img = img[:, 0:1, :, :]  # Select the first channel for all images in the batch
        elif img.ndim == 3:  # Single image in (C, H, W) format
            img = img[0:1, :, :]  # Select the first channel
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.show()

    # Assuming 'data_loader_train' is your DataLoader
    data_loader_iter = iter(data_loader_train)
    images, labels = next(data_loader_iter)

    # Display the first image's first channel and its label
    print(f"Label: {labels[0]}")
    imshow(images[0])  # Pass the first image of the batch

    for batch_idx, (inputs, targets) in enumerate(data_loader_train):
        print(f"Batch {batch_idx}:")
        print(f"Inputs: {inputs}, Targets: {targets}")
        # Optionally, break after the first batch to just see one example
        break
    
    #end diagnostics MMMMMMMMMMMMMMMMMMMM