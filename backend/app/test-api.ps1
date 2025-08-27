# FeelOri API Testing Script - Simple Version

$baseUrl = "http://localhost:8000/api/v1"
$password = "feeloriSuperSecure2024"

Write-Host "FeelOri API Testing Script" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green

# Test API Health
Write-Host ""
Write-Host "Testing API health..." -ForegroundColor Yellow

try {
    $healthResponse = Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET
    Write-Host "API is responding!" -ForegroundColor Green
} catch {
    Write-Host "API health check failed" -ForegroundColor Red
}

# Login and get access token
Write-Host ""
Write-Host "Logging in..." -ForegroundColor Yellow

$loginBody = @{
    password = $password
} | ConvertTo-Json

try {
    $loginResponse = Invoke-WebRequest -Uri "$baseUrl/auth/login" -Method POST -ContentType "application/json" -Body $loginBody
    
    if ($loginResponse.StatusCode -eq 200) {
        $tokenData = $loginResponse.Content | ConvertFrom-Json
        $accessToken = $tokenData.access_token
        Write-Host "Login successful!" -ForegroundColor Green
        Write-Host "Token expires in: $($tokenData.expires_in) seconds" -ForegroundColor Gray
        
        # Test authenticated endpoint
        Write-Host ""
        Write-Host "Testing /auth/me endpoint..." -ForegroundColor Yellow
        
        $headers = @{
            "Authorization" = "Bearer $accessToken"
            "Content-Type" = "application/json"
        }
        
        try {
            $meResponse = Invoke-WebRequest -Uri "$baseUrl/auth/me" -Method GET -Headers $headers
            
            if ($meResponse.StatusCode -eq 200) {
                $userData = $meResponse.Content | ConvertFrom-Json
                Write-Host "Authentication test successful!" -ForegroundColor Green
                Write-Host "User: $($userData.data.user.username)" -ForegroundColor Gray
            }
        } catch {
            Write-Host "Authentication test failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        # Display information
        Write-Host ""
        Write-Host "Available Endpoints:" -ForegroundColor Cyan
        Write-Host "  Login:     POST $baseUrl/auth/login" -ForegroundColor Gray
        Write-Host "  User Info: GET  $baseUrl/auth/me" -ForegroundColor Gray
        Write-Host "  Logout:    POST $baseUrl/auth/logout" -ForegroundColor Gray
        Write-Host "  Dashboard: GET  http://localhost:8000/static/dashboard.html" -ForegroundColor Gray
        
        Write-Host ""
        Write-Host "Your Access Token:" -ForegroundColor Cyan
        Write-Host $accessToken -ForegroundColor Yellow
        
    } else {
        Write-Host "Login failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "Login error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test complete!" -ForegroundColor Green