##' Runner follows the Part_0-Part8 .py files closely. Use the Part_#.py CLI arguments to form similar configurations and pipelines


## ---CENTRALIZED RUNS--- ##
./tools/run_centralized_base_V3.zsh


## ---DECENTRALIZED FL RUN WITH FULL HEs PSIs TLS--- ##
./MMFD_SYS_WRAP.zsh --psi oprf --he ckks --tls tls --zeroize none --clients 2 --rounds 1 && \
./MMFD_SYS_WRAP.zsh --psi oprf --he paillier --tls tls --zeroize none --clients 2 --rounds 1 && \
./MMFD_SYS_WRAP.zsh --psi hmac --he ckks --tls tls --zeroize none --clients 2 --rounds 1 && \
./MMFD_SYS_WRAP.zsh --psi hmac --he paillier --tls tls --zeroize none --clients 2 --rounds 1 && \
./MMFD_SYS_WRAP.zsh --psi oprf --he none --tls tls --zeroize none --clients 2 --rounds 1 && \
./MMFD_SYS_WRAP.zsh --psi hmac --he none --tls tls --zeroize none --clients 2 --rounds 1 && \
./MMFD_SYS_WRAP.zsh --psi oprf --he ckks --tls tls --zeroize zeroization --clients 2 --rounds 1



