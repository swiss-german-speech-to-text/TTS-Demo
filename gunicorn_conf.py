import multiprocessing

bind = "127.0.0.1:80"
# 1 worke need approx 3 GB of memory
# The mem_limit: is set to 8G in the docker-compose
workers = 2
# workers = multiprocessing.cpu_count() * 2 + 1