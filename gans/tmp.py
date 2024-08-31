# Move models to GPU
discriminator = discriminator.to("cuda")
generator = generator.to("cuda")

# Training loop
for epoch in range(num_epochs):
    for real_data in train_dataloader:
        real_labels = torch.ones(len(real_data), 1).to("cuda")
        fake_labels = torch.zeros(len(real_data), 1).to("cuda")

        real_data = real_data.float().view(real_data.size(0), -1).to("cuda")  # Move real data to GPU

        real_output = discriminator(real_data)
        d_loss_real = criterion(real_output, real_labels)
        real_score = real_output

        z = torch.randn(len(real_data), input_size_generator).unsqueeze(1).repeat(1, 855, 1).to("cuda")  # Generate random noise on GPU
        fake_data = generator(z)
        fake_data = fake_data.view(fake_data.size(0), -1)  # Flatten the fake data
        fake_data = fake_data.to("cuda")  # Move fake data to GPU

        fake_output = discriminator(fake_data)
        d_loss_fake = criterion(fake_output, fake_labels)
        fake_score = fake_output

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        z = torch.randn(len(real_data), input_size_generator).unsqueeze(1).repeat(1, 855, 1).to("cuda")  # Generate random noise on GPU
        fake_data = generator(z)
        output = discriminator(fake_data.view(fake_data.size(0), -1))  # Flatten the fake data

        g_loss = criterion(output, real_labels)

        # Add the length penalty loss
        length_penalty = length_penalty_loss(fake_data, len(real_data))
        total_g_loss = g_loss + lambda_penalty * length_penalty

        g_optimizer.zero_grad()
        total_g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {total_g_loss.item():.4f}, length_penalty: {length_penalty.item():.4f}')

# After training, move generated data back to CPU for plotting
fake_data = fake_data.cpu()

# Example usage:
plot_segments_from_json(generated_data)




class FCGenerator(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size, dropout_prob=0.8):
        super(FCGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.dropout(self.relu(self.fc1(x)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.dropout(self.relu(self.fc3(out)))
        out = self.dropout(self.relu(self.fc4(out)))
        out = self.fc5(out)
        return out