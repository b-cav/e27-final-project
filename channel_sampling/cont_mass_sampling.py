"""
mass_sampling.py

ENGS 27 Final Project Mass HTTP Sampling
Group - Ben Sheppard, Joshua Johnson, Ben Cavanagh

The following code interfaces with the noisy channel, hosted on a Dartmouth website.
"""

import asyncio
import aiohttp
import csv

# Semaphore to limit maximum concurrent requests
# Set very low to guard against crashes but slower
sem = asyncio.Semaphore(1000)

# Asynchronous single HTTP POST request function given target URL and data
async def post_req(session, url, code, writer, file) :
    async with sem :
        async with session.post(url, data = {"bits": code}) as response :
            output = await response.text()
            result = (output.split("<body>")[1].split("</body>")[0])
            writer.writerow([result])
            file.flush()  # Insta-write to file

# Asynchronous per-codeword request bundling
# Launch post requests for a given codeword n_req times
async def launch_task(session, url, code, n_reqs) :
    file = open(f"./cont_results_{n_reqs//1000}K/{code}_results.csv", "w", newline = "")
    writer = csv.writer(file)
    await asyncio.gather(
        *[post_req(session, url, code, writer, file) for _ in range(n_reqs)]
    )
    file.close()
    return code

# Create a task for each codeword, gather the results, and write to CSV
async def mass_request(n_reqs: int) :
    #codes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    codes = ["0"*100, "1"*100, "01"*50, "10"*50]
    print(codes)
    channel_url = "http://10.135.171.65:8080"
    async with aiohttp.ClientSession() as session :
        task_list = [asyncio.create_task(launch_task(session, channel_url, code, n_reqs)) for code in codes]

        for task in asyncio.as_completed(task_list) :
            code = await task
            print(f"Finished {code}")

if __name__ == "__main__" :
    asyncio.run(mass_request(1000000))

