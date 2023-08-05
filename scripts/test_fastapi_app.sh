#!/bin/bash
curl -X POST http://0.0.0.0:8080/v1/inference --header "Content-Type: application/json" -d '{"someInput": "asdf"}'