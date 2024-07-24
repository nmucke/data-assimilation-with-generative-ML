import torch






def compute_maximum_a_posteriori(
    latent_vec,
    log_prob,
    num_iterations=500,        
    lr=5e-2
):
    latent_vec.requires_grad = True
    optimizer = torch.optim.LBFGS([latent_vec], lr=lr, line_search_fn='strong_wolfe')

    # L-BFGS
    def closure():
        optimizer.zero_grad()
        loss = -log_prob(latent_vec)
        loss.backward()
        return loss

    for i in range(num_iterations):
        # optimizer.zero_grad()
        # loss = -log_prob(latent_vec)
        # loss.backward()
        optimizer.step(closure)
        loss = -log_prob(latent_vec)
        # optimizer.step()

        if i % 1 == 0:
            print(f'Loss: {loss.item()}, Iteration: {i}')

    return latent_vec


