#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = false;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.57;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  NIS_radar_ = 0;
  NIS_laser_ = 0;
  // P - Matrix
  n_aug_ = 7;
  n_x_ = 5;
  lambda_ = 3 - n_aug_;
  R_laser_ = MatrixXd(2, 2);
  R_laser_.fill(0.0);
  R_laser_(0,0) = std_laspx_ * std_laspx_;
  R_laser_(1,1) = std_laspy_ * std_laspy_;
  
  R_radar_ = MatrixXd(3, 3);
  R_radar_.fill(0.0);
  R_radar_(0,0) = std_radr_ * std_radr_;
  R_radar_(1,1) = std_radphi_ * std_radphi_;
  R_radar_(2,2) = std_radrd_ * std_radrd_;
  
  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  MeasurementPackage::SensorType sensorType = meas_package.sensor_type_;
  if (!is_initialized_) {
    const VectorXd raw = meas_package.raw_measurements_;
    previous_timestamp_ = meas_package.timestamp_;
    time_us_ = meas_package.timestamp_;
    if (sensorType == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double r = raw[0];
      double rho = raw[1];
      double px = r * cos(rho);
      double py = r * sin(rho);
      x_ << px, py, 0, 0, 0;
    }
    else if (sensorType == MeasurementPackage::LASER) {
      x_ << raw[0], raw[1], 0, 0, 0;
    }
    P_.fill(0.0);
    P_(0,0) = 0.3;
    P_(1,1) = 0.3;
    // velocity, yaw angle and yaw rate are unknown
    P_(2,2) = 1;
    P_(3,3) = 1;
    P_(4,4) = 1;
    
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    is_initialized_ = true;
  } else {
    if (sensorType == MeasurementPackage::LASER && use_laser_) {
      double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1e6;
      previous_timestamp_ = meas_package.timestamp_;
      if (delta_t >= 0.001) {
        Prediction(delta_t);
      }
      // cout << "-------" << endl;
      // cout << x_ << endl << P_ << endl;
      UpdateLidar(meas_package);
    } else if (sensorType == MeasurementPackage::RADAR && use_radar_) {
      double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1e6;
      previous_timestamp_ = meas_package.timestamp_;
      if (delta_t >= 0.001) {
        Prediction(delta_t);
      }
      UpdateRadar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.`
 */
void UKF::Prediction(double delta_t) {
  int cols = 2 * n_aug_ + 1;
  int rows = n_aug_;
  MatrixXd Xsig_aug = MatrixXd(rows, cols);
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;
  P_aug.topLeftCorner ( n_x_, n_x_ ) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;
  
  MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  MatrixXd Sigma = sqrt(lambda_ + n_aug_) * A;
  
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + Sigma.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - Sigma.col(i);
  }

  Xsig_pred_.fill(0.0);
  double delta_t2 = delta_t * delta_t;
  //predict sigma points
  VectorXd xpred(n_x_);
  for (int i = 0; i < Xsig_aug.cols(); i++) {
    VectorXd xk = Xsig_aug.col(i);
    double psi = xk(3);
    double psidot = xk(4);
    double v = xk(2);
    double nu_a = xk(5);
    double nu_psidotdot = xk(6);
    
    if (fabs(psidot) < 0.001) {
      xpred(0) = xk(0) + v * cos(psi) * delta_t;
      xpred(1) = xk(1) + v * sin(psi) * delta_t;
    } else {
      xpred(0) = xk(0) + v * (+sin(psi + psidot * delta_t) - sin(psi)) / psidot;
      xpred(1) = xk(1) + v * (-cos(psi + psidot * delta_t) + cos(psi)) / psidot;
    }
    
    xpred(2) = xk(2) + 0;
    xpred(3) = xk(3) + psidot * delta_t;
    xpred(4) = xk(4) + 0;
    
    // add noise
    xpred(0) += 0.5 * delta_t2 * cos(psi) * nu_a;
    xpred(1) += 0.5 * delta_t2 * sin(psi) * nu_a;
    xpred(2) += delta_t * nu_a;
    xpred(3) += 0.5 * delta_t2 * nu_psidotdot;
    xpred(4) += delta_t * nu_psidotdot;
    
    Xsig_pred_.col(i) = xpred;
  }

  //cout << "XSig_pred" << endl << Xsig_pred_ << endl;
  // -- Calculate updated state and covariance
  VectorXd weights = VectorXd(2*n_aug_+1);
  x_.fill(0.0);
  P_.fill(0.0);
  //set weights
  for (int i = 0; i < weights.size(); i++) {
    double w = 1 / (lambda_ + n_aug_);
    if (i == 0) {
      weights(i) = w * lambda_;
    } else {
      weights(i) = w * 0.5;
    }
    x_ = x_ + weights(i) * Xsig_pred_.col(i);
  }
  for (int i = 0; i < weights.size(); i++) {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    P_ = P_ + (weights(i) * diff * diff.transpose());
  }
  cout << "Predicted mean, cov\n" << x_ << endl << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd weights = VectorXd(2*n_aug_+1);
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  VectorXd z_pred(n_z);
  z_pred.fill(0.0);
  VectorXd z_actual = meas_package.raw_measurements_;
  
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd xi = Xsig_pred_.col(i);
    VectorXd zi(n_z);
    zi(0) = xi(0);
    zi(1) = xi(1);
    Zsig.col(i) = zi;
  }
  cout << "\nZsig\n" << Zsig << endl;
  
  for (int i = 0; i < weights.size(); i++) {
    double w = 1 / (lambda_ + n_aug_);
    if (i == 0) {
      weights(i) = w * lambda_;
    } else {
      weights(i) = w * 0.5;
    }
  }
  
  z_pred.fill(0.0);
  //calculate mean predicted measurement
  for (int i = 0; i < weights.size(); i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }
  cout << "\nZpred\n" << z_pred << endl;
  
  //calculate measurement covariance matrix S
  for (int i = 0; i < weights.size(); i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    S = S + (weights(i) * diff * diff.transpose());
  }
  S = S + R_laser_;
  
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }
  printMatrix(Tc, "TC --");
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //residual
  VectorXd z_diff = z_actual - z_pred;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  while (x_(3)> M_PI) x_(3)-=2.*M_PI;
  while (x_(3)<-M_PI) x_(3)+=2.*M_PI;
  cout << "Updated mean, cov\n" << x_ << endl << P_ << "\n---------\n";
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd weights = VectorXd(2*n_aug_+1);
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  VectorXd z_pred(n_z);
  z_pred.fill(0.0);
  VectorXd z_actual = meas_package.raw_measurements_;
  
  //transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd xi = Xsig_pred_.col(i);
    VectorXd zi(n_z);
    double px = xi(0);
    double py = xi(1);
    double v = xi(2);
    double psi = xi(3);
    
    zi(0) = sqrt(px * px + py * py);
    zi(1) = atan2(py, px);
    zi(2) = (px * cos(psi) * v + py * sin(psi) * v) / zi(0);
    
    Zsig.col(i) = zi;
    
  }
  
  for (int i = 0; i < weights.size(); i++) {
    double w = 1 / (lambda_ + n_aug_);
    if (i == 0) {
      weights(i) = w * lambda_;
    } else {
      weights(i) = w * 0.5;
    }
  }
  
  //calculate mean predicted measurement
  for (int i = 0; i < weights.size(); i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  for (int i = 0; i < weights.size(); i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + (weights(i) * z_diff * z_diff.transpose());
  }
  S = S + R_radar_;
  
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }
  //Kalman gain K;
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc * Sinv;
  
  //residual
  VectorXd z_diff = z_actual - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  
  x_ = x_ + K * z_diff;
  while (x_(3)> M_PI) x_(3)-=2.*M_PI ;
  while (x_(3)<-M_PI) x_(3)+=2.*M_PI;

  P_ = P_ - K * S * K.transpose();
}

void UKF::printMatrix (const MatrixXd& m, const char* name) {
  cout << endl << name << endl << m << endl;
}
