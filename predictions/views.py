
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.conf import settings
import json
import requests
from .forms import PatientDataForm, UserUpdateForm
from .models import Patient, Prediction, Assessment, LabSearch
from django.views.decorators.csrf import csrf_exempt
from django.contrib.admin.views.decorators import staff_member_required

from ml_model.predict import HeartDiseasePredictor
import os
import io
from .utils import generate_medical_pdf
import pandas as pd
import PyPDF2

# Initialize predictor
predictor = HeartDiseasePredictor()

from django.db.models import Avg, Count

@login_required
def home(request):
    # Summary Metrics - per-user when authenticated, otherwise global
    if request.user.is_authenticated:
        user_preds = Prediction.objects.filter(user=request.user)
        total_predictions = user_preds.count()
        patient_qs = Patient.objects.filter(predictions__user=request.user).distinct()
        total_patients = patient_qs.count()
        avg_age = patient_qs.aggregate(Avg('age'))['age__avg'] or 0

        # Heart Disease Prevalence (user scope)
        has_disease = user_preds.filter(prediction_result=1).count()
        no_disease = user_preds.filter(prediction_result=0).count()
        prevalence = (has_disease / total_predictions * 100) if total_predictions > 0 else 0

        # Gender Distribution (user scope)
        gender_stats = patient_qs.values('sex').annotate(count=Count('sex'))
        gender_labels = ['Male' if s['sex'] == 1 else 'Female' for s in gender_stats]
        gender_values = [s['count'] for s in gender_stats]

        # Age Groups Distribution (user scope)
        age_groups = {
            '18-30': patient_qs.filter(age__gte=18, age__lte=30).count(),
            '31-45': patient_qs.filter(age__gt=30, age__lte=45).count(),
            '46-60': patient_qs.filter(age__gt=45, age__lte=60).count(),
            '60+': patient_qs.filter(age__gt=60).count(),
        }

        # Prepare recent predictions with formatted probability
        recent_qs = user_preds.select_related('patient').order_by('-predicted_at')[:5]
        recent_predictions = []
        for p in recent_qs:
            p.probability_display = f"{p.probability:.2f}"
            recent_predictions.append(p)
    else:
        # Global metrics for anonymous visitors
        total_patients = Patient.objects.count()
        avg_age = Patient.objects.aggregate(Avg('age'))['age__avg'] or 0
        total_predictions = Prediction.objects.count()

        has_disease = Prediction.objects.filter(prediction_result=1).count()
        no_disease = Prediction.objects.filter(prediction_result=0).count()
        prevalence = (has_disease / total_predictions * 100) if total_predictions > 0 else 0

        gender_stats = Patient.objects.values('sex').annotate(count=Count('sex'))
        gender_labels = ['Male' if s['sex'] == 1 else 'Female' for s in gender_stats]
        gender_values = [s['count'] for s in gender_stats]

        age_groups = {
            '18-30': Patient.objects.filter(age__gte=18, age__lte=30).count(),
            '31-45': Patient.objects.filter(age__gt=30, age__lte=45).count(),
            '46-60': Patient.objects.filter(age__gt=45, age__lte=60).count(),
            '60+': Patient.objects.filter(age__gt=60).count(),
        }

        # Prepare recent predictions with formatted probability
        recent_qs = Prediction.objects.select_related('patient').order_by('-predicted_at')[:5]
        recent_predictions = []
        for p in recent_qs:
            p.probability_display = f"{p.probability:.2f}"
            recent_predictions.append(p)

    context = {
        'metrics': {
            'total_patients': total_patients,
            'avg_age': round(avg_age, 1),
            'prevalence': round(prevalence, 1),
            'total_predictions': total_predictions,
        },
        'charts': {
            'disease_ratio': json.dumps([no_disease, has_disease]),
            'gender': json.dumps({'labels': gender_labels, 'values': gender_values}),
            'age_dist': json.dumps({'labels': list(age_groups.keys()), 'values': list(age_groups.values())}),
        },
        'recent_activity': recent_predictions
    }
    return render(request, 'predictions/home.html', context)

def register_view(request):
    from .forms import RegistrationForm
    from django.core.mail import send_mail
    from django.conf import settings

    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()            # Create audit log entry about the new registration
            from .models import AuditLog
            try:
                AuditLog.objects.create(action='USER_CREATED', performed_by=None, target_user=user, details=f"Registered with email {user.email}")
            except Exception as e:
                print(f"Failed to create audit log: {e}")            # Notify staff users about pending approval
            staff_emails = list(User.objects.filter(is_staff=True).values_list('email', flat=True))
            subject = 'New user registration awaiting approval'
            message = f'User {user.username} has registered and requires validation. Visit admin to approve.'
            if staff_emails:
                try:
                    send_mail(subject, message, getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@example.com'), staff_emails, fail_silently=True)
                except Exception as e:
                    print(f"Failed to send admin notification email: {e}")

            messages.success(request, 'Registration received — an administrator will review and approve your account. You will receive an email when approved.')
            return redirect('predictions:login')
    else:
        form = RegistrationForm()
    # Add Bootstrap classes
    for field in form.fields.values():
        if hasattr(field.widget, 'attrs'):
            field.widget.attrs['class'] = 'form-control'
    return render(request, 'predictions/register.html', {'form': form})

@login_required
def predict_view(request):
    if request.method == 'POST':
        form = PatientDataForm(request.POST)
        if form.is_valid():
            # Save patient data
            patient = form.save(commit=False)
            patient.patient_name = form.cleaned_data.get('patient_name') or "My Self"
            patient.save()
            
            # Prepare data for prediction
            data = form.cleaned_data
            
            try:
                # Make prediction
                result, probability = predictor.predict(data)
                
                # Save prediction
                prediction = Prediction.objects.create(
                    user=request.user,
                    patient=patient,
                    prediction_result=result,
                    probability=probability
                )
                
                messages.success(request, 'Prediction calculated successfully.')
                # Prepare friendly probability values for template
                probability_value = float(getattr(prediction, 'probability', 0.0) or 0.0)
                prob_percent = int(round(probability_value * 100))
                prob_display = f"{probability_value * 100:.2f}%"

                return render(request, 'predictions/predict.html', {
                    'form': form,
                    'prediction': prediction,
                    'patient': patient,
                    'prob_percent': prob_percent,
                    'prob_display': prob_display
                })
                
            except Exception as e:
                messages.error(request, f'Error during prediction: {str(e)}')
                return render(request, 'predictions/predict.html', {'form': form})
    else:
        form = PatientDataForm()
    
    return render(request, 'predictions/predict.html', {'form': form})

@login_required
def history_view(request):
    selected_patient = request.GET.get('patient', 'All')

    try:
        # Fetch all types of history
        predictions = Prediction.objects.filter(user=request.user).select_related('patient').order_by('-predicted_at')
        assessments = Assessment.objects.filter(user=request.user).order_by('-taken_at')
        lab_searches = LabSearch.objects.filter(user=request.user).order_by('-searched_at')

        # Get unique patient names for filter
        names_set = set()
        for p in predictions:
            try:
                if p.patient and getattr(p.patient, 'patient_name', None):
                    names_set.add(p.patient.patient_name)
            except Exception as e:
                print(f"Warning: skipping patient name due to error: {e}")
        names_set.update(a.patient_name for a in assessments if getattr(a, 'patient_name', None))
        names_set.update(l.patient_name for l in lab_searches if getattr(l, 'patient_name', None))
        patient_names = sorted(list(names_set))

        # Filter if selected
        if selected_patient != 'All':
            predictions = predictions.filter(patient__isnull=False, patient__patient_name=selected_patient)
            assessments = assessments.filter(patient_name=selected_patient)
            lab_searches = lab_searches.filter(patient_name=selected_patient)

        # Create unified timeline items (defensive and verbose on error)
        timeline = []
        for p in predictions:
            try:
                if p is None:
                    continue
                date = getattr(p, 'predicted_at', None)
                # Safe result display
                try:
                    result = p.get_prediction_result_display()
                except Exception:
                    result = str(getattr(p, 'prediction_result', 'Unknown'))
                # Safe probability formatting
                prob_val = getattr(p, 'probability', 0.0) or 0.0
                try:
                    probability = f"{prob_val*100:.1f}%"
                except Exception:
                    probability = str(prob_val)
                patient = getattr(p, 'patient', None)
                # Build a friendly name using first/last if available, otherwise fallback to patient_name
                if patient:
                    fname = getattr(patient, 'first_name', None) or ''
                    lname = getattr(patient, 'last_name', None) or ''
                    if fname or lname:
                        name = f"{fname} {lname}".strip()
                    else:
                        name = getattr(patient, 'patient_name', 'Unknown')
                    dob = getattr(patient, 'birth_date', None)
                else:
                    name = 'Unknown'
                    dob = None

                timeline.append({
                    'type': 'prediction',
                    'date': date,
                    'title': 'Heart Disease Prediction',
                    'result': result,
                    'probability': probability,
                    'obj': p,
                    'name': name,
                    'dob': dob
                })
            except Exception as e:
                print(f"Error building timeline entry for prediction {getattr(p, 'id', 'unknown')}: {e}")
                continue

        for a in assessments:
            try:
                timeline.append({
                    'type': 'assessment',
                    'date': getattr(a, 'taken_at', None),
                    'title': 'Stress & Wellness Test',
                    'result': f"{getattr(a, 'risk_level', 'Unknown')} Risk",
                    'score': f"PHQ:{getattr(a, 'score_phq', '?')}, GAD:{getattr(a, 'score_gad', '?')}",
                    'obj': a,
                    'name': getattr(a, 'patient_name', 'Unknown')
                })
            except Exception as e:
                print(f"Error building timeline entry for assessment {getattr(a, 'id', 'unknown')}: {e}")
                continue

        for l in lab_searches:
            try:
                timeline.append({
                    'type': 'lab',
                    'date': getattr(l, 'searched_at', None),
                    'title': 'Medicine Search',
                    'query': getattr(l, 'query', ''),
                    'results': getattr(l, 'results_count', 0),
                    'obj': l,
                    'name': getattr(l, 'patient_name', 'Unknown')
                })
            except Exception as e:
                print(f"Error building timeline entry for lab search {getattr(l, 'id', 'unknown')}: {e}")
                continue

        # Sort timeline by date descending; guard missing dates
        from django.utils import timezone
        import datetime
        min_date = timezone.make_aware(datetime.datetime.min.replace(year=1900))
        timeline = sorted(timeline, key=lambda x: x.get('date') or min_date, reverse=True)

        return render(request, 'predictions/history_v2.html', {
            'timeline': timeline,
            'predictions_count': predictions.count(),
            'assessments_count': assessments.count(),
            'lab_count': lab_searches.count(),
            'patient_names': patient_names,
            'selected_patient': selected_patient
        })
    except Exception as e:
        import traceback
        print(f"Error in history_view: {e}")
        print(traceback.format_exc())
        messages.error(request, 'Unable to load history due to an internal error. Please try again later.')
        return render(request, 'predictions/history.html', {
            'timeline': [],
            'predictions_count': 0,
            'assessments_count': 0,
            'lab_count': 0,
            'patient_names': [],
            'selected_patient': selected_patient
        })

@login_required
def patient_detail(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    predictions = patient.predictions.all()
    return render(request, 'predictions/patient_detail.html', {
        'patient': patient, 
        'predictions': predictions
    })

@login_required
def chatbot_view(request):
    return render(request, 'predictions/chatbot.html')

def chatbot_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')
        
        try:
            # Call External AI API (DeepSeek/OpenAI Compatible)
            headers = {
                "Authorization": f"Bearer {settings.CHATBOT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat", # Default DeepSeek model
                "messages": [
                    {"role": "system", "content": "You are CardioAI, a professional medical assistant specialized in cardiovascular diseases. Provide helpful, accurate, and empathetic information about heart health, symptoms, prevention, and treatment. Always remind users to consult a real doctor for medical emergencies."},
                    {"role": "user", "content": user_message}
                ],
                "stream": False
            }
            
            response = requests.post(settings.CHATBOT_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result['choices'][0]['message']['content']
            else:
                bot_response = f"I'm sorry, I am having trouble connecting to my AI brain right now (Error: {response.status_code}). Please try again later."
                
        except Exception as e:
            bot_response = "I encountered an error while processing your request. Please check your internet connection and try again."
            print(f"Chatbot API Error: {str(e)}")

        return JsonResponse({'response': bot_response})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def trends_view(request):
    import pandas as pd
    import os
    from django.conf import settings
    import json

    csv_path = os.path.join(settings.BASE_DIR, 'data', 'health_trends.csv')
    
    # Default stats if file missing
    context = {'national': '{}', 'age': '{}', 'sex': '{}'}
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # 1. National Trend (Over Time periods)
        national_df = df[df['Group'] == 'National Estimate'].sort_values('Time Period')
        national_data = {
            'labels': national_df['Time Period Label'].tolist(),
            'values': national_df['Value'].tolist()
        }
        
        # 2. Breakdown by Age (Latest Period)
        latest_period = df['Time Period'].max()
        age_df = df[(df['Group'] == 'By Age') & (df['Time Period'] == latest_period)]
        age_data = {
            'labels': age_df['Subgroup'].tolist(),
            'values': age_df['Value'].tolist()
        }
        
        # 3. Breakdown by Sex
        sex_df = df[(df['Group'] == 'By Sex') & (df['Time Period'] == latest_period)]
        # Filter for simple Male/Female to avoid complexity in this first view
        sex_df = sex_df[sex_df['Subgroup'].isin(['Male', 'Female'])]
        sex_data = {
            'labels': sex_df['Subgroup'].tolist(),
            'values': sex_df['Value'].tolist()
        }
        
        context = {
            'national': json.dumps(national_data),
            'age': json.dumps(age_data),
            'sex': json.dumps(sex_data),
        }

    return render(request, 'predictions/trends.html', context)

@login_required
def assessment_view(request):
    return render(request, 'predictions/assessment.html')

@login_required
def assessment_result(request):
    if request.method == 'POST':
        try:
            # PHQ-2 Questions
            phq1 = int(request.POST.get('phq1', 0))
            phq2 = int(request.POST.get('phq2', 0))
            
            # Additional Wellness Questions
            phq3 = int(request.POST.get('phq3', 0)) # Sleep
            phq4 = int(request.POST.get('phq4', 0)) # Energy
            
            # GAD-2 Questions
            gad1 = int(request.POST.get('gad1', 0))
            gad2 = int(request.POST.get('gad2', 0))
            
            # Lifestyle Questions (0-3 scale)
            phys1 = int(request.POST.get('phys1', 0)) # Exercise
            phys2 = int(request.POST.get('phys2', 0)) # Diet
            
            has_symptoms = (phq1 + phq2 >= 3) or (gad1 + gad2 >= 3)
            risk_level = "High" if has_symptoms else "Low"
            
            # Calculate Scores
            total_phq = phq1 + phq2 + phq3 + phq4
            total_gad = gad1 + gad2
            wellness_score = phys1 + phys2
            
            # Compare with Data
            import pandas as pd
            import os
            from django.conf import settings
            
            csv_path = os.path.join(settings.BASE_DIR, 'data', 'health_trends.csv')
            compare_val = 19.4 # Default national
            subgroup_label = "national average"
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                latest_period = df['Time Period'].max()
                
                match_subgroup = "Experienced symptoms of anxiety/depression in past 4 weeks" if has_symptoms else "Did not experience symptoms of anxiety/depression in the past 4 weeks"
                trend_match = df[(df['Subgroup'] == match_subgroup) & (df['Time Period'] == latest_period)]
                
                if not trend_match.empty and not pd.isna(trend_match.iloc[0]['Value']):
                    compare_val = float(trend_match.iloc[0]['Value'])
                    subgroup_label = "people reporting similar symptoms in recent national surveys"
                else:
                    national_match = df[(df['Group'] == 'National Estimate') & (df['Time Period'] == latest_period)]
                    if not national_match.empty and not pd.isna(national_match.iloc[0]['Value']):
                        compare_val = float(national_match.iloc[0]['Value'])
                        subgroup_label = "national average (latest available)"

            # Save to Database
            Assessment.objects.create(
                user=request.user,
                score_phq=total_phq,
                score_gad=total_gad,
                wellness_score=wellness_score,
                risk_level=risk_level
            )

            context = {
                'risk_level': risk_level,
                'has_symptoms': has_symptoms,
                'compare_val': compare_val,
                'subgroup_label': subgroup_label,
                'score_phq': phq1 + phq2,
                'full_phq': total_phq,
                'score_gad': total_gad,
                'wellness_score': wellness_score,
            }
            return render(request, 'predictions/assessment_result.html', context)
        except (ValueError, TypeError):
            return redirect('predictions:assessment')
            
    return redirect('predictions:assessment')

@login_required
def lab_upload(request):
    if request.method == 'POST':
        user_description = request.POST.get('description', '').lower()
        pdf_file = request.FILES.get('medical_file')
        
        if not pdf_file and not user_description:
            messages.error(request, 'Please provide a description OR upload a PDF file to start.')
            return redirect('predictions:lab_upload')
            
        try:
            extracted_text = ""
            if pdf_file:
                # Extract text from PDF
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    extracted_text += (page.extract_text() or "")
            
            search_query = (user_description + " " + extracted_text.lower()).strip()
            
            if not search_query:
                messages.error(request, 'No readable content found in PDF. Please provide a description.')
                return redirect('predictions:lab_upload')
            
            # Load medicines dataset
            med_path = os.path.join(settings.BASE_DIR, 'data', 'medicines_dataset.csv')
            if not os.path.exists(med_path):
                messages.error(request, 'Medicine dataset not found.')
                return redirect('predictions:lab_upload')
                
            med_df = pd.read_csv(med_path)
            
            results = []
            # Basic keyword matching scoring
            for _, row in med_df.iterrows():
                score = 0
                # Match against Indication, Category, and Name
                indication = str(row['Indication']).lower()
                category = str(row['Category']).lower()
                name = str(row['Name']).lower()
                
                # Check for direct keyword matches
                for word in search_query.split():
                    if len(word) < 3: continue # Skip short words
                    if word in indication: score += 5
                    if word in category: score += 3
                    if word in name: score += 2
                
                if score > 0:
                    results.append({
                        'name': row['Name'],
                        'category': row['Category'],
                        'dosage_form': row['Dosage Form'],
                        'strength': row['Strength'],
                        'manufacturer': row['Manufacturer'],
                        'indication': row['Indication'],
                        'classification': row['Classification'],
                        'score': score
                    })
            
            # Sort by score and take top matches
            results = sorted(results, key=lambda x: x['score'], reverse=True)[:15]
            
            # Log the search
            LabSearch.objects.create(
                user=request.user,
                patient_name=request.POST.get('patient_name') or "My Self",
                query=user_description,
                results_count=len(results)
            )
            
            return render(request, 'predictions/lab_results.html', {
                'results': results,
                'query': user_description
            })
            
        except Exception as e:
            messages.error(request, f'Error processing PDF: {str(e)}')
            return redirect('predictions:lab_upload')
            
    return render(request, 'predictions/lab_upload.html')

@login_required
def profile_update(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        if u_form.is_valid():
            u_form.save()
            messages.success(request, f'Your profile has been updated!')
            return redirect('predictions:profile_update')
    else:
        u_form = UserUpdateForm(instance=request.user)
        
    context = {
        'u_form': u_form
    }
    return render(request, 'predictions/profile_update.html', context)

@login_required
def roadmap_view(request):
    # Roadmap feature has been intentionally removed per project decision.
    # Returning 410 GONE to indicate the endpoint is no longer available.
    return HttpResponse('Roadmap has been removed', status=410)
@login_required
def simulator_view(request):
    return render(request, 'predictions/simulator.html')

@csrf_exempt
@login_required
def api_simulate_risk(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # data structure matches predictor.predict expectations
            result, probability = predictor.predict(data)
            return JsonResponse({
                'result': 'Heart Disease Detected' if result == 1 else 'No Disease Detected',
                'probability': float(probability),
                'risk_level': 'Critical' if probability > 0.7 else 'Moderate' if probability > 0.4 else 'Low'
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'POST required'}, status=400)

@login_required
def export_report_pdf(request):
    predictions = Prediction.objects.filter(user=request.user).select_related('patient').order_by('-predicted_at')
    assessments = Assessment.objects.filter(user=request.user).order_by('-taken_at')
    lab_searches = LabSearch.objects.filter(user=request.user).order_by('-searched_at')
    
    pdf_buffer = generate_medical_pdf(request.user, predictions, assessments, lab_searches)
    
    response = HttpResponse(pdf_buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="CardioAI_Report_{request.user.username}.pdf"'
    return response

@csrf_exempt
@login_required
def api_check_interaction(request):
    # This endpoint has been removed. Return 410 GONE to indicate the functionality is intentionally removed.
    return JsonResponse({'error': 'Drug Interaction Checker has been removed'}, status=410)

@staff_member_required
def ai_health_check(request):
    """Run quick checks against the configured AI endpoints and return diagnostics."""
    provider = getattr(settings, 'CHATBOT_PROVIDER', 'deepseek').lower()
    results = []

    if provider == 'gemini':
        urls = []
        base = settings.CHATBOT_API_URL
        urls.append(base)
        if 'v1beta2' in base:
            urls.append(base.replace('v1beta2', 'v1'))
        elif 'v1' in base and 'v1beta2' not in base:
            urls.append(base.replace('v1', 'v1beta2'))
        urls.append('https://generativelanguage.googleapis.com/v1/models/text-bison-001:generateText')

        for url in urls:
            try:
                headers = {"Content-Type": "application/json"}
                if getattr(settings, 'CHATBOT_GEMINI_USE_API_KEY', True) and settings.CHATBOT_API_KEY:
                    sep = '&' if '?' in url else '?'
                    test_url = f"{url}{sep}key={settings.CHATBOT_API_KEY}"
                    resp = requests.post(test_url, headers=headers, json={"prompt": {"text": "Hello"}}, timeout=20)
                else:
                    headers["Authorization"] = f"Bearer {settings.CHATBOT_API_KEY}"
                    resp = requests.post(url, headers=headers, json={"prompt": {"text": "Hello"}}, timeout=20)
                results.append({'url': url, 'status': resp.status_code, 'body': resp.text[:1000]})
            except Exception as e:
                results.append({'url': url, 'error': str(e)})

    else:
        # Basic check for DeepSeek-style endpoint
        url = settings.CHATBOT_API_URL
        try:
            headers = {"Authorization": f"Bearer {settings.CHATBOT_API_KEY}", "Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json={"model": "deepseek-chat", "messages": [{"role": "system", "content": "Ping"}]}, timeout=20)
            results.append({'url': url, 'status': resp.status_code, 'body': resp.text[:1000]})
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return JsonResponse({'provider': provider, 'results': results})


@login_required
def pending_registrations(request):
    if not request.user.is_superuser:
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("You do not have permission to view this page.")
    """Staff page to review and approve/reject pending user registrations."""
    from django.core.mail import send_mail
    from django.conf import settings
    from django.utils import timezone
    from django.contrib.auth.models import User
    
    pending_users = User.objects.filter(is_active=False, profile__is_validated=False).select_related('profile')

    if request.method == 'POST':
        action = request.POST.get('action')
        selected = request.POST.getlist('selected')
        approved = 0
        rejected = 0
        for uid in selected:
            try:
                user = User.objects.get(pk=int(uid))
                profile = getattr(user, 'profile', None)
                if action == 'approve':
                    if profile:
                        profile.is_validated = True
                        profile.validated_by = request.user
                        profile.validated_at = timezone.now()
                        profile.save()
                    user.is_active = True
                    user.save()
                    # Send approval email
                    try:
                        send_mail(
                            'Your CardioAI account has been approved',
                            'Your account has been validated by an administrator. You can now log in.',
                            getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@example.com'),
                            [user.email],
                            fail_silently=True
                        )
                    except Exception as e:
                        print(f"Failed to send approval email to {user.email}: {e}")
                    # Audit log
                    from .models import AuditLog
                    AuditLog.objects.create(action='USER_APPROVED', performed_by=request.user, target_user=user, details='Approved via staff page')
                    approved += 1
                elif action == 'reject':
                    if profile:
                        profile.is_validated = False
                        profile.validated_by = request.user
                        profile.validated_at = timezone.now()
                        profile.save()
                    user.is_active = False
                    user.save()
                    try:
                        send_mail(
                            'Your CardioAI registration was not approved',
                            'Your registration was reviewed and not approved. Contact support for more info.',
                            getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@example.com'),
                            [user.email],
                            fail_silently=True
                        )
                    except Exception as e:
                        print(f"Failed to send rejection email to {user.email}: {e}")
                    from .models import AuditLog
                    AuditLog.objects.create(action='USER_REJECTED', performed_by=request.user, target_user=user, details='Rejected via staff page')
                    rejected += 1
            except Exception as e:
                print(f"Error processing user id {uid}: {e}")
                continue

        if approved:
            messages.success(request, f"{approved} user(s) approved")
        if rejected:
            messages.warning(request, f"{rejected} user(s) rejected")
        return redirect('predictions:pending_registrations')

    return render(request, 'predictions/pending_registrations.html', {'pending_users': pending_users})