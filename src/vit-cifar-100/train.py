import torch
from tqdm import tqdm


def train(
    net,
    device,
    trainloader,
    testloader,
    HYPERPARAMS,
    criterion,
    optimizer,
    wandb_run,
    scheduler,
    log_interval=2000,
    eval_interval=5,
):
    net.train()
    for epoch in range(HYPERPARAMS["epochs"]):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Training Epoch: {epoch+1}",
        ):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                accuracy = 100.0 * correct / total
                for param_group in optimizer.param_groups:
                    current_lr = param_group["lr"]
                tqdm.write(
                    f"[Epoch {epoch+1}, Batch {i}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "batch": i + 1,
                        "train_loss": avg_loss,
                        "train_accuracy": accuracy,
                        "learning_rate": current_lr,
                    }
                )
                running_loss = 0.0
                correct = 0
                total = 0

        scheduler.step()

        if (epoch + 1) % eval_interval == 0:
            test_accuracy = test(
                net, testloader, device, wandb_run, criterion=criterion
            )
            tqdm.write(f"[Epoch {epoch+1}] Test Accuracy: {test_accuracy:.2f}%")

    print("Finished Training")


def test(net, testloader, device, wandb_run, criterion):
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = 100 * correct / total

    tqdm.write(f"Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
    wandb_run.log({"test_accuracy": accuracy, "test_loss": avg_loss})

    return accuracy
