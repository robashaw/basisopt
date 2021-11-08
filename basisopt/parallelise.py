from distributed import LocalCluster, Client, wait

def chunk(x, n_chunks):
    chunk_size = len(x) // n_chunks
    remainder = len(x) % n_chunks
    chunk_list = [chunk_size]*n_chunks
    for i in range(remainder):
        chunk_list[i] += 1
    new_x = []
    ctr = 0
    for i in range(n_chunks):
        new_x.append(x[ctr:ctr+chunk_list[i]])
        ctr += chunk_list[i]
    return new_x

def distribute(n_proc, func, x, **kwargs):
    n_chunks = len(x) // n_proc
    if len(x) % n_proc > 0:
        n_chunks += 1
    new_x = chunk(x, n_chunks)

    all_results = []
    for i in range(n_chunks):
        cluster = LocalCluster(n_workers=len(new_x[i]), processes=True)
        client = Client(cluster)
        ens = client.map(func, new_x[i], **kwargs)
        wait(ens)
        results = [e.result() for e in ens]
        all_results.append(results)
        client.close()
        cluster.close()
    
    return [item for sublist in all_results for item in sublist]
