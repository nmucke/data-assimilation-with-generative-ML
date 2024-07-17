import torch






def compute_maximum_a_posteriori(
    latent_vec,
    sampling_model,
    observation_operator,
    log_prob,
    num_iterations=500,        
    lr=5e-2
):
    latent_vec.requires_grad = True
    optimizer = torch.optim.Adam([latent_vec], lr=lr)

    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = -log_prob(latent_vec)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Loss: {loss.item()}, Iteration: {i}')

    return latent_vec


