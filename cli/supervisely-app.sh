#!/bin/bash

set -o pipefail -e

VERSION='1.0.2'

usage() {
  echo -e "Supervisely Apps CLI

Usage:
  supervisely-app release

Options:
  -s | --server  server address to release
  -t | --token   api token
  -p | --path    path to app (current dir by default)
  -a | --sub-app path to app folder inside multi-app repository (relative to --path / current dir)

Commands:
  release\t\t\t release app release
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
    -a | --sub-app)
            rel_submodule_path="${2}"
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
    echo -e "curl is not installed. Please install it and try again\n"
  fi

  if ! command -v tar > /dev/null 2>&1; then
    failed=1
    echo -e "tar is not installed. Please install it and try again\n"
  fi

  if [[ ${failed} -eq 1 ]]; then
    exit ${failed}
  fi
}

function release() {
  if [[ -z "${module_path}" ]]; then
    echo "No '--path' is provided, will archive and release the current directory"
    module_path=$(pwd)
  fi

  module_path=${module_path%/}
  module_root=${module_path}

  if [[ -n "${rel_submodule_path}" ]]; then
    rel_submodule_path=${rel_submodule_path%/}
    module_path="${module_path}/${rel_submodule_path}"

    echo "App from subfolder ${rel_submodule_path} will be released"
  fi


  if [[ -f ~/supervisely.env ]]; then
    echo "Detected ~/supervisely.env"
    source ~/supervisely.env
  fi

  if [[ -z "${server}" ]]; then
    if [[ -n "${SERVER_ADDRESS}" ]]; then
      server=$(echo "${SERVER_ADDRESS}" | sed 's/\r$//')
    else
      echo -e "\"server\" is not specified. Please use -s to set server"
      exit 1
    fi
  fi

  # trim traliling slash if any
  server=${server%/}

  if [[ -z "${token}" ]]; then
    if [[ -n "${API_TOKEN}" ]]; then
      token=$(echo "${API_TOKEN}" | sed 's/\r$//')
    else
      echo -e "\"token\" is not specified. Please use -t to set token"
      exit 1
    fi
  fi

  if [ ! -f "${module_path}/config.json" ]; then
    echo -e "\"config.json\" not found in ${module_path}"
    exit 1
  fi

  modal_template=
  config=$(cat "${module_path}/config.json")
  archive_path="/tmp/$(echo $RANDOM$RANDOM$RANDOM | tr '[0-9]' '[a-z]')"
  modal_template_path=$(echo "${config}" | sed -nE 's/"modal_template": "(.*)",?/\1/p' | xargs)
  parsed_slug=
  parsed_slug_config=$(echo "${config}" | sed -nE 's/"slug": "(.*)",?/\1/p' | xargs)
  module_name=$(echo "${config}" | sed -nE 's/^ *"name": "(.*)",?/\1/p' | xargs)
  module_release=$(echo "${config}" | sed -nE 's/"release": ({.*})/\1/p' | xargs)

  if [[ "${parsed_slug_config}" ]]; then
    parsed_slug="${parsed_slug_config}"

    if [[ -n "${rel_submodule_path}" ]]; then
      if [[ "${parsed_slug_config}" != *"/${rel_submodule_path}" ]]; then
        echo "Slug from submodule config.json must includes submodule path (specified in -a | --sub-app)"
        exit 1
      fi
    fi

    # echo "Application slug in config.json: ${parsed_slug}"
  fi

  if [[ -z "${parsed_slug}" ]]; then
    echo "Slug is empty. Please add slug field in config.json"
    exit 1
  fi

  if [[ -f "${module_path}/README.md" ]]; then
    readme=$(cat "${module_path}/README.md")
  fi

  if [[ -n "${modal_template_path}" ]]; then
    modal_template=$(cat "${module_root}/${modal_template_path}")
  fi

  module_exists_label="updated"

  exists_status=$(curl -w '%{http_code}' -sS --output /dev/null -L --location --request POST "${server}/public/api/v3/ecosystem.info" \
    --header "x-api-key: ${token}" \
    --header "Content-Type: application/json" \
    -d '{"slug": "'"${parsed_slug}"'"}');

  if [[ "$exists_status" =~ ^4 ]]; then
    module_exists_label="created"
  fi

  echo
  echo "App \"${module_name}\" will be ${module_exists_label}"
  echo "Slug: ${parsed_slug}"
  echo "Release: ${module_release}"
  echo "Server: ${server}"
  echo "Local path: ${module_root}"

  if [[ -n "${rel_submodule_path}" ]]; then
    echo "Submodule path: ${rel_submodule_path}"
  fi

  echo "Do you want to continue? [y/N]"
  read -n 1 -r response
  echo

  if ! [[ $response =~ ^[Yy]$ ]]
  then
    echo "Release canceled"
    exit 1
  fi

  mkdir "${archive_path}"

  echo "Packing the following files to ${archive_path}/archive.tar.gz:"

  if [ -f "${module_root}/.gitignore" ] && command -v git > /dev/null 2>&1; then
    echo "$(git ls-files -c --others --exclude-standard)"

    git_files=($(git ls-files -c --others --exclude-standard))
    files_list=$(printf "$(basename $module_root)/%s " "${git_files[@]}")

    tar -czf "$archive_path/archive.tar.gz" -C "$(dirname $module_root)" ${files_list}
  else
    tar -v --exclude-vcs --totals -czf "$archive_path/archive.tar.gz" -C "$(dirname $module_root)" $(basename $module_root)
  fi

  echo "Uploading archive..."

  curl_params=()
  if [[ -f "${module_path}/README.md" ]]; then
    curl_params+=(-F readme="<${module_path}/README.md")
  fi

  release_response=$(curl "${curl_params[@]}" --http1.1 -L --location --request POST "${server}/public/api/v3/ecosystem.release" \
    --progress-bar \
    --header "x-api-key: ${token}" \
    -F slug="${parsed_slug}" \
    -F config="${config}" \
    -F archive=@"$archive_path/archive.tar.gz" \
    --form-string modalTemplate="${modal_template}" | cat)

  if [[ "$release_response" =~ '{"success":true}' ]]; then
    echo "Application ${parsed_slug} successfully released to ${server}"
  else
    echo "ERROR: $release_response"
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
