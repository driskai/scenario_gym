#!/bin/sh

# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpgtar --decrypt --directory ./ --gpg-args="--passphrase=$EXAMPLE_XOSC_PASSKEY --batch --quiet --yes" ./tests/input_files/example_xoscs.gpg