#!/bin/bash

VERSION='1.0.0'

usage() {
  echo -e "Supervisely Apps CLI

Usage:
  supervisely-apps release

Options:
  -s | --server server address to release
  -t | --token  api token
  -p | --path   path to app (current dir by default)

Commands:
  release\t\t\t release app version
  ";
}

print_version() {
  echo "Version: ${VERSION}"
}

while test $# -gt 0; do
  case "${1}" in
    -h|--help)
            usage
            exit 0
            shift
            ;;
    -v|--version)
            print_version
            exit 0
            shift
            ;;
    -s|--server)
            server="${2}"
            shift
            shift
            ;;
    -t|--token)
            token="${2}"
            shift
            shift
            ;;
    -p|--path)
            module_path="${2}"
            shift
            shift
            ;;
    *)
            if [[ -z "${action}" ]]; then
              action="${1}"
            else
              args+=("${1}")
            fi

            shift
            ;;
  esac
done


check_cli_deps() {
  local failed=0

  if ! command -v curl > /dev/null 2>&1; then
    failed=1
    echo -e "curl is not installed. Please run this command to install it:\nsudo apt update && sudo apt install -y curl\n"
  fi

  if ! command -v tar > /dev/null 2>&1; then
    failed=1
    echo -e "tar is not installed. Please run this command to install it:\nsudo apt update && sudo apt install -y tar\n"
  fi

  if [[ ${failed} -eq 1 ]]; then
    exit ${failed}
  fi
}

function release() {
  if [[ -z "${module_path}" ]]; then
    module_path=$(pwd)
  fi

  if [[ -f ~/supervisely.env ]]; then
    source ~/supervisely.env
  fi

  if [[ -z "${server}" ]]; then
    if [[ -n "${SERVER_ADDRESS}" ]]; then
      server=$SERVER_ADDRESS
    else
      echo -e "\"server\" is not specified. Please use -s to set server"
      exit 1
    fi
  fi

  if [[ -z "${token}" ]]; then
    if [[ -n "${API_TOKEN}" ]]; then
      token=$API_TOKEN
    else
      echo -e "\"token\" is not specified. Please use -t to set token"
      exit 1
    fi
  fi

  if [ ! -f "${module_path}/config.json" ]; then
    echo -e "\"config.json\" not found in ${module_path}"
    exit 1
  fi

  readme=
  modal_template=
  config=$(cat "${module_path}/config.json")
  archive_path="/tmp/$(echo $RANDOM | md5sum | head -c 20; echo;)"
  modal_template_path=$(echo "${config}" | sed -n 's/"modal_template": "\(.*\)",\?/\1/p' | xargs)

  if [[ -f "${module_path}/README.md" ]]; then
    readme=$(cat "${module_path}/README.md")
  fi

  if [[ -n "${modal_template_path}" ]]; then
    modal_template=$(cat "${module_path}/${modal_template_path}")
  fi

  mkdir "${archive_path}"
  
  tar -czf "$archive_path/archive.tar.gz" -C "$(dirname $module_path)" $(basename $module_path)

  release_response=$(curl -sSL -w '%{http_code}' --location --request POST "${server}/public/api/v3/ecosystem.release" \
  --header "x-api-key: ${token}" \
  -F config="${config}" \
  -F readme="${readme}" \
  -F archive=@"$archive_path/archive.tar.gz" \
  --form-string modalTemplate="${modal_template}"
  )

  if [[ "$release_response" =~ '{"success":true}200' ]]; then
    echo "Module successfully released"
  else
    echo "ERROR: server returned HTTP code $release_response"
    exit 1
  fi

  rm -rf $archive_path
}

function main() {
  check_cli_deps

  if [[ "${action}" == 'release' ]]; then
    release
    exit 0
  fi

  usage
  exit 1
}

main
