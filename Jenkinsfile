def Pipeline(String DockerImg, String DockerRunArgs, String GpuType) {
    docker.image("${DockerImg}").inside("${DockerRunArgs}") {
        sh '''
            set -e
            rsync -a --delete --exclude 'Paddle/' --exclude 'backends/musa/third_party/' /workspace/ /home/paddle_musa/
        '''

        gitlabCommitStatus(name: "01-${GpuType}-env prepare", state: "running") {
            sh '''
                set -e
                cd /home/paddle_musa

                apt-get update && apt-get install -y patchelf

                pip3 install -r ./Paddle/python/requirements.txt
  
                pip3 install -r ./python/requirements.txt
            '''
        }

        gitlabCommitStatus(name: "02-${GpuType}-build paddle", state: "running") {
            sh '''
                set -e
                cd /home/paddle_musa

                EXPECTED_PADDLE_COMMIT=$(git rev-parse HEAD:Paddle)
                CURRENT_PADDLE_COMMIT=$(git -C Paddle rev-parse HEAD)
                echo "Expected Paddle submodule commit: ${EXPECTED_PADDLE_COMMIT}"
                echo "Current Paddle repository commit: ${CURRENT_PADDLE_COMMIT}"

                if [ "${CURRENT_PADDLE_COMMIT}" != "${EXPECTED_PADDLE_COMMIT}" ]; then
                    echo "Paddle commit mismatch, reset Paddle repository and checkout expected commit."
                    cd Paddle

                    echo "Paddle status before reset:"
                    git status --short

                    git reset --hard HEAD
                    git clean -fd
                    git submodule update --init --recursive
                    git submodule foreach --recursive 'git reset --hard HEAD'
                    git submodule foreach --recursive 'git clean -fd'

                    git pull --ff-only || git fetch --all --tags --prune
                    git checkout "${EXPECTED_PADDLE_COMMIT}"
                    git submodule update --init --recursive

                    echo "Paddle status after checkout:"
                    git status --short

                    cd /home/paddle_musa
                fi

                cd /home/paddle_musa/backends/musa
                echo y | bash tools/build.sh -c
                echo y | bash tools/build.sh -ap
                echo y | bash tools/build.sh -u

                pip3 install numpy==1.23.5
            '''
        }

        gitlabCommitStatus(name: "03-${GpuType}-unit test", state: "running") {
            sh '''
                set -e
                cd /home/paddle_musa
                export PADDLE_XCCL_BACKEND=musa
                export MUSA_VISIBLE_DEVICES=6,7
                cd ./backends/musa && bash tools/run_ut.sh
            '''
        }
    }
}

pipeline {
  agent none

  options {
    gitLabConnection('sh-code')
  }

  environment {
    S5000IMG = 'sh-harbor.mthreads.com/mt-ai/musa-paddle-dev:rc4.3.4-20260525-ci-ph1'
    DOCKER_RUN_ARGS = '--network=host ' +
      '--user root ' +
      '--privileged ' +
      '--shm-size 20G ' +
      '--pid=host ' +
      '-e MTHREADS_VISIBLE_DEVICES=all ' +
      '-e MUSA_VISIBLE_DEVICES=all ' +
      '-v $WORKSPACE:/workspace'
  }

  stages {
    stage('Run task in parallel') {
      parallel {
        stage('paddle_musa') {
          agent { label 'paddle_musa' }
          steps {
            deleteDir()
            checkout scm
            timeout(time: 200, unit: 'MINUTES') {
              script {
                Pipeline("${S5000IMG}", "${DOCKER_RUN_ARGS}", "S5000")
              }
            }
          }
        }
      }
    }
  }

  post {
    unstable {
      script {
        currentBuild.result = 'FAILURE'
        error("Build marked as FAILURE due to instability.")
      }
      updateGitlabCommitStatus name: '06-final', state: 'failed'
    }
    failure {
      updateGitlabCommitStatus name: '06-final', state: 'failed'
    }
    success {
      updateGitlabCommitStatus name: '06-final', state: 'success'
    }
    aborted {
      updateGitlabCommitStatus name: '06-final', state: 'canceled'
    }
  }
}
