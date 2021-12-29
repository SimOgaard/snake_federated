FROM nvcr.io/nvidia/pytorch:21.03-py3

ARG u_id
ARG g_id
ARG username

RUN groupadd --gid ${g_id} ${username}
RUN useradd --uid ${u_id} --gid ${g_id} --shell /bin/bash --create-home ${username}
USER ${username}
RUN chown -R ${u_id}:${g_id} /home/${username}
RUN chmod -R  755 /home/${username}

# ALT 1 (comment if you want to do ALT 2)
RUN git clone https://github.com/SimOgaard/snake_federated

WORKDIR /snake_federated