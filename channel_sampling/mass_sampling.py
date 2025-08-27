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
sem = asyncio.Semaphore(10)

# Asynchronous single HTTP POST request function given target URL and data
async def post_req(session, url, code) :
    async with sem :
        async with session.post(url, data = {"bits": code}) as response :
            output = await response.text()
            return (output.split("<body>")[1].split("</body>")[0])

# Asynchronous per-codeword request bundling
# Launch post requests for a given codeword n_req times
async def launch_task(session, url, code, n_reqs) :
    results = await asyncio.gather(
        *[post_req(session, url, code) for _ in range(n_reqs)]
    )
    return code, results

# Create a task for each codeword, gather the results, and write to CSV
async def mass_request(n_reqs: int) :
    #codes = ["0", "1", "00", "01", "10", "11", "000", "001", "010", "011", "100", "101", "110", "111"]
    codes = ["0"*10, "1"*10, "01"*5, "10"*5]
    print(codes)
    channel_url = "https://engs27.host.dartmouth.edu/cgi-bin/noisychannel.py"
    async with aiohttp.ClientSession() as session :
        task_list = [asyncio.create_task(launch_task(session, channel_url, code, n_reqs)) for code in codes]

        for task in asyncio.as_completed(task_list) :
            code, results = await task
            file = open(f"./results100K/{code}_results.csv", "w", newline = "")
            # Writes all at once so memory-inefficient for large requests, but stable
            # enough and we just needed to get our sample sets
            writer = csv.writer(file)
            for result in results :
                writer.writerow([result])

if __name__ == "__main__" :
    asyncio.run(mass_request(10000))

