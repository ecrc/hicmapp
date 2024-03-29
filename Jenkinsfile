pipeline {
    agent { label 'jenkinsfile' }
    triggers {
        pollSCM('H/10 * * * *')
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('mkl') {
            stages {
                stage ('build without StarPu-MPI') {
                    steps {
                        sh '''#!/bin/bash -le
                            ####################################################
                            # Configure and build
                            ####################################################
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            module load hwloc/2.4.0-gcc-10.2.0
                            ####################################################
                            # BLAS/LAPACK
                            ####################################################
                            module load mkl/2020.0.166
                            ####################################################
                            set -x
                            git submodule update --init --recursive
                            ./config.sh -t -e
                            ./clean_build.sh
                        '''
                    }
                }
                stage ('test without StarPu-MPI') {
                    steps {

                        sh '''#!/bin/bash -le
                            ####################################################
                            # Run tester
                            ####################################################
                            echo "========================================"
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            module load hwloc/2.4.0-gcc-10.2.0
                            ####################################################
                            # BLAS/LAPACK
                            ####################################################
                            module load mkl/2020.0.166
                            cd bin/
                            ctest --no-compress-output --verbose
                            '''
                    }
                }
                stage ('build with StarPu-MPI') {
                    steps {
                        sh '''#!/bin/bash -le
                            ####################################################
                            # Configure and build
                            ####################################################
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            module load hwloc/2.4.0-gcc-10.2.0
                            ####################################################
                            # BLAS/LAPACK/MPI
                            ####################################################
                            source /opt/ecrc/hpc-toolkit/ub18/setvars.sh
                            ####################################################
                            set -x
                            git submodule update --init --recursive
                            ./config.sh -t -e -r starpu -m
                            ./clean_build.sh
                        '''
                    }
                }
                stage ('test with StarPu-MPI') {
                    steps {

                        sh '''#!/bin/bash -le
                            ####################################################
                            # Run tester
                            ####################################################
                            echo "========================================"
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            module load hwloc/2.4.0-gcc-10.2.0
                            ####################################################
                            # BLAS/LAPACK/MPI
                            ####################################################
                            source /opt/ecrc/hpc-toolkit/ub18/setvars.sh
                            ####################################################
                            cd bin/
                            ctest --no-compress-output --verbose
                            '''
                    }
                }
            }
        }
        	stage('documentation') {
                     agent { label 'jenkinsfile'}
                     steps {
                         sh '''#!/bin/bash -le
                            module purge
                            module load gcc/10.2.0
                            module load cmake/3.21.2
                            ####################################################
                            # BLAS/LAPACK
                            ####################################################
                            module load mkl/2020.0.166
                            ./config.sh -t -e
                            ./clean_build.sh
                            cd bin
                            make docs
                            '''
                         publishHTML( target: [allowMissing: false, alwaysLinkToLastBuild: false, keepAll: false, reportDir: 'docs/html', reportFiles: 'index.html', reportName: 'Doxygen Documentation', reportTitles: ''] )
                     }
                }
    }

    // Post build actions
    post {
        //always {
        //}
        //success {
        //}
        //unstable {
        //}
        //failure {
        //}
        unstable {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build is UNSTABLE", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
        failure {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build FAILED", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
    }
}
