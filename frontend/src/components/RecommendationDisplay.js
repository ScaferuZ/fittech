import React, { useState, useEffect } from 'react';
import {
  Box,
  Stack,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  SimpleGrid,
  Badge,
  Progress,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useColorModeValue,
  Divider,
  Card,
  CardBody,
  CardHeader,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  List,
  ListItem,
  ListIcon,
  Icon,
  Flex,
  Spacer,
  Skeleton,
  SkeletonText,
  SkeletonCircle
} from '@chakra-ui/react';
import { CheckCircleIcon, InfoIcon, WarningIcon } from '@chakra-ui/icons';
import { nutritionService } from '../services/nutritionService';
import { mealPlanService } from '../services/mealPlanService';
import { apiService } from '../services/api';

const RecommendationDisplay = ({ recommendations, userData, onBack, onNewRecommendation, onMealPlanGenerated }) => {
  const [nutritionData, setNutritionData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mealPlan, setMealPlan] = useState(null);
  const [mealPlanLoading, setMealPlanLoading] = useState(true);
  const [backendMealPlan, setBackendMealPlan] = useState(null);
  const [backendMealPlanLoading, setBackendMealPlanLoading] = useState(true);

  const gradientBg = useColorModeValue(
    'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)'
  );

  // Calculate user's daily macro targets based on API response and template data
  const calculateDailyMacros = () => {
    if (!recommendations || !userData) return null;

    // Try to get nutrition data from various possible locations in the API response
    const nutrition = recommendations?.nutrition_recommendation || 
                     recommendations?.predictions?.nutrition_template ||
                     recommendations?.nutrition_template;
    
    if (!nutrition) return null;

    const weight = parseFloat(userData.weight);
    const userProfile = recommendations?.user_profile || {};
    const tdee = userProfile.tdee || userProfile.total_daily_energy_expenditure || 2000;
    
    // Debug logging to see what data we're getting
    console.log('🔍 Frontend Debug - Data Sources:', {
      weight,
      tdee,
      nutrition_recommendation: recommendations?.nutrition_recommendation,
      predictions_nutrition: recommendations?.predictions?.nutrition_template,
      nutrition_template: recommendations?.nutrition_template,
      selected_nutrition: nutrition
    });
    
    // Priority 1: Use pre-calculated values from the API if available
    if (nutrition.target_calories && nutrition.target_protein && nutrition.target_carbs && nutrition.target_fat) {
      console.log('✅ Using pre-calculated API values:', {
        calories: nutrition.target_calories,
        protein: nutrition.target_protein,
        carbs: nutrition.target_carbs,
        fat: nutrition.target_fat
      });
      return {
        calories: Math.round(nutrition.target_calories),
        protein: Math.round(nutrition.target_protein),
        carbs: Math.round(nutrition.target_carbs),
        fat: Math.round(nutrition.target_fat)
      };
    }
    
    // Priority 2: Use template multipliers if available
    if (nutrition.caloric_intake_multiplier && nutrition.protein_per_kg && nutrition.carbs_per_kg && nutrition.fat_per_kg) {
      const calories = Math.round(tdee * nutrition.caloric_intake_multiplier);
      const protein = Math.round(weight * nutrition.protein_per_kg);
      const carbs = Math.round(weight * nutrition.carbs_per_kg);
      const fat = Math.round(weight * nutrition.fat_per_kg);
      
      console.log('✅ Using template multipliers:', {
        caloric_multiplier: nutrition.caloric_intake_multiplier,
        protein_per_kg: nutrition.protein_per_kg,
        carbs_per_kg: nutrition.carbs_per_kg,
        fat_per_kg: nutrition.fat_per_kg,
        calculated: { calories, protein, carbs, fat }
      });
      
      return { calories, protein, carbs, fat };
    }
    
    console.log('⚠️ Using fallback calculations');
    
    // Fallback: Use standard macro calculations for different goals (last resort)
    let calories, protein, carbs, fat;
    
    if (userData.fitness_goal === 'Fat Loss') {
      calories = Math.round(tdee * 0.8); // 20% deficit
      protein = Math.round(weight * 2.3); // Use template 1 value
      carbs = Math.round(weight * 1.8); // Use template 1 value
      fat = Math.round(weight * 1.0); // Use template 1 value
    } else if (userData.fitness_goal === 'Muscle Gain') {
      calories = Math.round(tdee * 1.1); // 10% surplus
      protein = Math.round(weight * 2.1); // Use template 5 value
      carbs = Math.round(weight * 4.25); // Use template 5 value
      fat = Math.round(weight * 1.0); // Use template 5 value
    } else { // Maintenance
      calories = Math.round(tdee * 0.95); // Use template 7 value
      protein = Math.round(weight * 1.8); // Use template 7 value
      carbs = Math.round(weight * 4.5); // Use template 7 value
      fat = Math.round(weight * 1.0); // Use template 7 value
    }

    return { calories, protein, carbs, fat };
  };

  // Load nutrition data from JSON file and generate meal plan
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load nutrition data
        const jsonData = await nutritionService.loadNutritionData();
        setNutritionData(jsonData);
        setLoading(false);

        // Generate meal plan if we have user data
        if (recommendations && userData) {
          setMealPlanLoading(true);
          setBackendMealPlanLoading(true);
          
          const dailyMacros = calculateDailyMacros();
          
          if (dailyMacros) {
            // Generate meal plan using the local service
            const mealPlanResult = await mealPlanService.generateDailyMealPlan(
              dailyMacros.calories,
              dailyMacros.protein,
              dailyMacros.carbs,
              dailyMacros.fat
            );

            if (mealPlanResult.success) {
              const transformedMealPlan = mealPlanService.transformMealPlanToFrontend(mealPlanResult);
              setMealPlan(transformedMealPlan);
            } else {
              console.warn('Failed to load organized meal plan, using fallback');
              setMealPlan(null);
            }
            setMealPlanLoading(false);

            // Also fetch meal plan from the backend API for comparison
            try {
              console.log('🔄 Fetching meal plan from backend with macros:', dailyMacros);
              const backendPlan = await apiService.getMealPlan(
                dailyMacros.calories,
                dailyMacros.protein,
                dailyMacros.carbs,
                dailyMacros.fat,
                { dietary_restrictions: [] } // Add preferences if needed
              );
              console.log('✅ Backend meal plan received:', backendPlan);
              setBackendMealPlan(backendPlan);
              
              // Store the detailed meal plan in the recommendation for future reference
              if (backendPlan && backendPlan.meal_plan && onMealPlanGenerated) {
                console.log('🔄 Storing meal plan for recommendation history...');
                onMealPlanGenerated(backendPlan.meal_plan);
              }
            } catch (backendError) {
              console.error('❌ Failed to fetch backend meal plan:', backendError);
              setBackendMealPlan(null);
            }
            setBackendMealPlanLoading(false);
          }
        }
      } catch (error) {
        console.error('Error loading data:', error);
        setLoading(false);
        setMealPlanLoading(false);
        setBackendMealPlanLoading(false);
      }
    };

    loadData();
  }, [recommendations, userData]);

  if (!recommendations) {
    return (
      <Box textAlign="center" py={10}>
        <Heading size="lg" mb={4}>🤔 Belum Ada Rekomendasi</Heading>
        <Text color="gray.600" mb={6}>
          Silakan isi formulir profil terlebih dahulu untuk mendapatkan rekomendasi yang dipersonalisasi.
        </Text>
        <Button colorScheme="brand" onClick={onBack}>
          Kembali ke Formulir
        </Button>
      </Box>
    );
  }

  // Map API response structure to component expected structure
  // Extract workout and nutrition data from the API response
  const workout = recommendations?.predictions?.workout_template;
  const nutrition = recommendations?.predictions?.nutrition_template;

  // Calculate food portions based on template requirements
  const calculateFoodPortions = (targetMacros) => {
    if (!nutritionData.length || !targetMacros) return [];

    const suggestions = [];
    
    // More balanced meal distribution
    const mealDistribution = {
      sarapan: { percentage: 0.25, name: '🌅 Sarapan', foodCount: 3 },
      makan_siang: { percentage: 0.40, name: '☀️ Makan Siang', foodCount: 4 },
      makan_malam: { percentage: 0.30, name: '🌙 Makan Malam', foodCount: 3 },
      camilan: { percentage: 0.05, name: '🍪 Camilan', foodCount: 1 }
    };

    Object.entries(mealDistribution).forEach(([mealType, config]) => {
      const targetCalories = targetMacros.calories * config.percentage;
      const targetProtein = targetMacros.protein * config.percentage;
      const targetCarbs = targetMacros.carbs * config.percentage;
      const targetFat = targetMacros.fat * config.percentage;
      
      // Select appropriate foods for this meal using nutrition service
      let selectedFoods = [];
      
      if (mealType === 'sarapan') {
        selectedFoods = nutritionService.getFoodsByCategory('breakfast');
      } else if (mealType === 'makan_siang') {
        selectedFoods = nutritionService.getFoodsByCategory('lunch');
      } else if (mealType === 'makan_malam') {
        selectedFoods = nutritionService.getFoodsByCategory('dinner');
      } else if (mealType === 'camilan') {
        selectedFoods = nutritionService.getFoodsByCategory('snack');
      }

      // If no specific foods found, use fallback selections
      if (selectedFoods.length === 0) {
        if (mealType === 'sarapan') {
          selectedFoods = nutritionData.filter(food => 
            food.name.toLowerCase().includes('telur') ||
            food.name.toLowerCase().includes('roti') ||
            food.name.toLowerCase().includes('mie') ||
            food.name.toLowerCase().includes('nasi')
          ).slice(0, 3);
        } else if (mealType === 'makan_siang') {
          selectedFoods = nutritionData.filter(food => 
            food.name.toLowerCase().includes('ayam') ||
            food.name.toLowerCase().includes('nasi') ||
            food.name.toLowerCase().includes('sayur') ||
            food.name.toLowerCase().includes('kentang')
          ).slice(0, 4);
        } else if (mealType === 'makan_malam') {
          selectedFoods = nutritionData.filter(food => 
            food.name.toLowerCase().includes('ikan') ||
            food.name.toLowerCase().includes('ayam') ||
            food.name.toLowerCase().includes('sayur') ||
            food.name.toLowerCase().includes('kentang')
          ).slice(0, 3);
        } else if (mealType === 'camilan') {
          selectedFoods = nutritionData.filter(food => 
            food.name.toLowerCase().includes('yogurt') ||
            food.name.toLowerCase().includes('buah') ||
            food.calories < 150
          ).slice(0, 1);
        }
        
        // Final fallback
        if (selectedFoods.length === 0) {
          selectedFoods = nutritionData.slice(0, config.foodCount);
        }
      }

      // Ensure we have enough foods for the meal
      if (selectedFoods.length < config.foodCount) {
        const additionalFoods = nutritionData.filter(food => 
          !selectedFoods.some(selected => selected.name === food.name)
        ).slice(0, config.foodCount - selectedFoods.length);
        selectedFoods = [...selectedFoods, ...additionalFoods];
      }

      // Calculate portions using a more balanced approach
      const mealFoods = selectedFoods.slice(0, config.foodCount).map((food, index) => {
        let baseCalories, finalGrams;
        
        if (mealType === 'camilan') {
          // Snacks should be smaller
          baseCalories = targetCalories;
          finalGrams = Math.min(100, Math.max(30, (baseCalories * 100) / food.calories));
        } else {
          // For main meals, distribute calories based on food type and position
          let calorieWeight = 1;
          
          // Give more calories to protein sources
          if (food.name.toLowerCase().includes('ayam') || 
              food.name.toLowerCase().includes('ikan') ||
              food.name.toLowerCase().includes('daging') ||
              food.name.toLowerCase().includes('telur')) {
            calorieWeight = 1.3;
          }
          // Less calories for vegetables
          else if (food.name.toLowerCase().includes('sayur') ||
                   food.name.toLowerCase().includes('tomat') ||
                   food.name.toLowerCase().includes('wortel')) {
            calorieWeight = 0.7;
          }
          // Moderate calories for carbs
          else if (food.name.toLowerCase().includes('nasi') ||
                   food.name.toLowerCase().includes('kentang') ||
                   food.name.toLowerCase().includes('roti')) {
            calorieWeight = 1.1;
          }
          
          baseCalories = (targetCalories / config.foodCount) * calorieWeight;
          finalGrams = Math.min(350, Math.max(50, (baseCalories * 100) / food.calories));
        }
        
        finalGrams = Math.round(finalGrams);
        
        return {
          ...food,
          grams: finalGrams,
          actualCalories: Math.round((food.calories / 100) * finalGrams),
          actualProtein: Math.round(((food.protein / 100) * finalGrams) * 10) / 10,
          actualCarbs: Math.round(((food.carbs / 100) * finalGrams) * 10) / 10,
          actualFat: Math.round(((food.fat / 100) * finalGrams) * 10) / 10
        };
      });

      // Adjust portions to better match targets
      const totalActualCalories = mealFoods.reduce((sum, food) => sum + food.actualCalories, 0);
      const calorieRatio = targetCalories / totalActualCalories;
      
      // Only adjust if the ratio is significantly off (more than 15% difference)
      if (Math.abs(calorieRatio - 1) > 0.15) {
        mealFoods.forEach(food => {
          const adjustedGrams = Math.round(food.grams * calorieRatio);
          food.grams = Math.min(mealType === 'camilan' ? 150 : 400, Math.max(30, adjustedGrams));
          food.actualCalories = Math.round((food.calories / 100) * food.grams);
          food.actualProtein = Math.round(((food.protein / 100) * food.grams) * 10) / 10;
          food.actualCarbs = Math.round(((food.carbs / 100) * food.grams) * 10) / 10;
          food.actualFat = Math.round(((food.fat / 100) * food.grams) * 10) / 10;
        });
      }

      suggestions.push({
        meal: config.name,
        targetCalories: Math.round(targetCalories),
        targetProtein: Math.round(targetProtein),
        targetCarbs: Math.round(targetCarbs),
        targetFat: Math.round(targetFat),
        actualCalories: mealFoods.reduce((sum, food) => sum + food.actualCalories, 0),
        actualProtein: Math.round(mealFoods.reduce((sum, food) => sum + food.actualProtein, 0) * 10) / 10,
        actualCarbs: Math.round(mealFoods.reduce((sum, food) => sum + food.actualCarbs, 0) * 10) / 10,
        actualFat: Math.round(mealFoods.reduce((sum, food) => sum + food.actualFat, 0) * 10) / 10,
        foods: mealFoods
      });
    });

    return suggestions;
  };

  // Transform backend meal plan to frontend format
  const transformBackendMealPlan = (backendPlan) => {
    if (!backendPlan || !backendPlan.daily_meal_plan) return [];

    const mealTypeMapping = {
      'sarapan': '🌅 Sarapan',
      'makan_siang': '☀️ Makan Siang', 
      'makan_malam': '🌙 Makan Malam',
      'snack': '🍪 Camilan'
    };

    const suggestions = [];

    Object.entries(backendPlan.daily_meal_plan).forEach(([mealType, mealData]) => {
      if (mealData && mealData.foods) {
        const foods = mealData.foods.map(food => ({
          name: food.nama,
          grams: food.amount,
          actualCalories: food.calories,
          actualProtein: food.protein,
          actualCarbs: food.carbs,
          actualFat: food.fat,
          calories: food.calories, // For compatibility
          protein: food.protein,
          carbs: food.carbs,
          fat: food.fat
        }));

        // Get meal targets from backend response
        const mealTargets = backendPlan.meal_targets && backendPlan.meal_targets[mealType];

        const targetCalories = mealTargets ? Math.round(mealTargets.calories) : Math.round(mealData.scaled_calories);
        const targetProtein = mealTargets ? Math.round(mealTargets.protein) : Math.round(mealData.scaled_protein);
        const targetCarbs = mealTargets ? Math.round(mealTargets.carbs) : Math.round(mealData.scaled_carbs);
        const targetFat = mealTargets ? Math.round(mealTargets.fat) : Math.round(mealData.scaled_fat);

        // Ensure we have valid numbers (fallback to scaled values if targets are missing)
        const finalTargetCalories = isNaN(targetCalories) ? Math.round(mealData.scaled_calories) : targetCalories;
        const finalTargetProtein = isNaN(targetProtein) ? Math.round(mealData.scaled_protein) : targetProtein;
        const finalTargetCarbs = isNaN(targetCarbs) ? Math.round(mealData.scaled_carbs) : targetCarbs;
        const finalTargetFat = isNaN(targetFat) ? Math.round(mealData.scaled_fat) : targetFat;

        suggestions.push({
          meal: mealTypeMapping[mealType] || mealType,
          mealName: mealData.meal_name,
          description: mealData.description,
          targetCalories: finalTargetCalories,
          targetProtein: finalTargetProtein,
          targetCarbs: finalTargetCarbs,
          targetFat: finalTargetFat,
          actualCalories: Math.round(mealData.scaled_calories),
          actualProtein: Math.round(mealData.scaled_protein * 10) / 10,
          actualCarbs: Math.round(mealData.scaled_carbs * 10) / 10,
          actualFat: Math.round(mealData.scaled_fat * 10) / 10,
          foods: foods
        });
      }
    });

    return suggestions;
  };

  const dailyMacros = calculateDailyMacros();
  
  // Use backend meal plan first (more accurate), then fallback to local plan, then calculated portions
  const foodSuggestions = backendMealPlan && backendMealPlan.daily_meal_plan 
    ? transformBackendMealPlan(backendMealPlan)
    : (mealPlan && mealPlan.length > 0 
      ? mealPlan 
      : (dailyMacros ? calculateFoodPortions(dailyMacros) : []));

  // Loading skeleton for the entire component
  if (loading) {
    return (
      <Box maxW={{ base: "full", lg: "6xl" }} mx="auto" p={{ base: 4, md: 8 }}>
        <VStack spacing={{ base: 6, md: 8 }} align="stretch">
          {/* Header Skeleton */}
          <Box textAlign="center" py={{ base: 4, md: 6 }}>
            <Skeleton height="40px" mb={2} />
            <Skeleton height="20px" width="60%" mx="auto" />
          </Box>

          {/* Profile Summary Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="200px" />
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
                {[...Array(6)].map((_, i) => (
                  <Box key={i}>
                    <Skeleton height="16px" width="80px" mb={1} />
                    <Skeleton height="24px" width="120px" />
                  </Box>
                ))}
              </SimpleGrid>
            </CardBody>
          </Card>

          {/* Metrics Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="250px" />
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, sm: 2, md: 4 }} spacing={{ base: 4, md: 6 }}>
                {[...Array(4)].map((_, i) => (
                  <Box key={i}>
                    <Skeleton height="16px" width="100px" mb={1} />
                    <Skeleton height="24px" width="80px" />
                  </Box>
                ))}
              </SimpleGrid>
            </CardBody>
          </Card>

          {/* Workout Section Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="300px" />
            </CardHeader>
            <CardBody>
              <VStack spacing={4}>
                <Skeleton height="100px" width="full" />
                <Skeleton height="80px" width="full" />
              </VStack>
            </CardBody>
          </Card>

          {/* Nutrition Section Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="280px" />
            </CardHeader>
            <CardBody>
              <VStack spacing={4}>
                <Skeleton height="120px" width="full" />
                <Skeleton height="200px" width="full" />
              </VStack>
            </CardBody>
          </Card>
        </VStack>
      </Box>
    );
  }

  return (
    <Box maxW={{ base: "full", lg: "6xl" }} mx="auto" p={{ base: 4, md: 8 }}>
      <VStack spacing={{ base: 6, md: 8 }} align="stretch">
        {/* Header */}
        <Box textAlign="center" py={{ base: 4, md: 6 }}>
          <Heading size={{ base: "lg", md: "xl" }} bgGradient={gradientBg} bgClip="text" mb={2}>
            🎯 Rekomendasi XGFitness Anda
          </Heading>
          <Text color="gray.600" fontSize={{ base: "md", md: "lg" }}>
            Rekomendasi yang dipersonalisasi berdasarkan profil dan tujuan Anda
          </Text>
        </Box>

        {/* User Profile Summary */}
        <Card>
          <CardHeader>
            <Heading size={{ base: "sm", md: "md" }}>👤 Profil Anda</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
              <Stat>
                <StatLabel>Usia</StatLabel>
                <StatNumber>{userData?.age} tahun</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Jenis Kelamin</StatLabel>
                <StatNumber>{userData?.gender === 'Male' ? 'Pria' : 'Wanita'}</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Tinggi</StatLabel>
                <StatNumber>{userData?.height} cm</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Berat</StatLabel>
                <StatNumber>{userData?.weight} kg</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Tujuan</StatLabel>
                <StatNumber>
                  {userData?.fitness_goal === 'Fat Loss' ? 'Membakar Lemak' :
                   userData?.fitness_goal === 'Muscle Gain' ? 'Menambah Massa Otot' :
                   userData?.fitness_goal === 'Maintenance' ? 'Mempertahankan Berat' : 
                   userData?.fitness_goal}
                </StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Tingkat Aktivitas</StatLabel>
                <StatNumber>
                  {userData?.activity_level === 'Low Activity' ? 'Aktivitas Rendah' :
                   userData?.activity_level === 'Moderate Activity' ? 'Aktivitas Sedang' :
                   userData?.activity_level === 'High Activity' ? 'Aktivitas Tinggi' :
                   userData?.activity_level}
                </StatNumber>
              </Stat>
            </SimpleGrid>
          </CardBody>
        </Card>

        {/* User Metrics from API */}
        {recommendations?.user_profile && (
          <Card>
            <CardHeader>
              <Heading size={{ base: "sm", md: "md" }}>📊 Analisis Tubuh Anda</Heading>
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, sm: 2, md: 4 }} spacing={{ base: 4, md: 6 }}>
                <Stat>
                  <StatLabel>BMI</StatLabel>
                  <StatNumber>{recommendations.user_profile.bmi?.toFixed(1)}</StatNumber>
                  <StatHelpText>
                    <Badge colorScheme={recommendations.user_profile.bmi_category === 'Normal' ? 'green' : 'orange'}>
                      {recommendations.user_profile.bmi_category === 'Underweight' ? 'Kurus' :
                       recommendations.user_profile.bmi_category === 'Normal' ? 'Normal' :
                       recommendations.user_profile.bmi_category === 'Overweight' ? 'Kelebihan Berat' :
                       recommendations.user_profile.bmi_category === 'Obese' ? 'Obesitas' :
                       recommendations.user_profile.bmi_category}
                    </Badge>
                  </StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel>BMR</StatLabel>
                  <StatNumber>{Math.round(recommendations.user_profile.bmr)} kkal</StatNumber>
                  <StatHelpText>Kalori Basal</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel>TDEE</StatLabel>
                  <StatNumber>{Math.round(recommendations.user_profile.tdee)} kkal</StatNumber>
                  <StatHelpText>Total Pengeluaran Energi</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel>Kategori BMI</StatLabel>
                  <StatNumber>
                    {recommendations.user_profile.bmi_category === 'Underweight' ? 'Kurus' :
                     recommendations.user_profile.bmi_category === 'Normal' ? 'Normal' :
                     recommendations.user_profile.bmi_category === 'Overweight' ? 'Kelebihan Berat' :
                     recommendations.user_profile.bmi_category === 'Obese' ? 'Obesitas' :
                     recommendations.user_profile.bmi_category}
                  </StatNumber>
                </Stat>
              </SimpleGrid>
            </CardBody>
          </Card>
        )}

        {/* Confidence Scores */}
        {recommendations.model_confidence && (
          <Card>
            <CardHeader>
              <Heading size={{ base: "sm", md: "md" }}>🎯 Tingkat Kepercayaan AI</Heading>
            </CardHeader>
            <CardBody>
              <VStack spacing={{ base: 4, md: 6 }} align="stretch">
                <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 4, md: 6 }}>
                  <Stat>
                    <StatLabel>Kepercayaan Keseluruhan</StatLabel>
                    <StatNumber color="green.500">
                      {Math.round(((recommendations.model_confidence.nutrition_confidence + recommendations.model_confidence.workout_confidence) / 2) * 100)}%
                    </StatNumber>
                    <StatHelpText>
                      <Badge colorScheme="green">Tinggi</Badge>
                    </StatHelpText>
                  </Stat>
                  <Stat>
                    <StatLabel>Kepercayaan Nutrisi</StatLabel>
                    <StatNumber color="blue.500">
                      {Math.round(recommendations.model_confidence.nutrition_confidence * 100)}%
                    </StatNumber>
                    <StatHelpText>
                      <StatArrow type={recommendations.model_confidence.nutrition_confidence > 0.5 ? 'increase' : 'decrease'} />
                    </StatHelpText>
                  </Stat>
                  <Stat>
                    <StatLabel>Kepercayaan Workout</StatLabel>
                    <StatNumber color="purple.500">
                      {Math.round(recommendations.model_confidence.workout_confidence * 100)}%
                    </StatNumber>
                    <StatHelpText>
                      <StatArrow type={recommendations.model_confidence.workout_confidence > 0.5 ? 'increase' : 'decrease'} />
                    </StatHelpText>
                  </Stat>
                </SimpleGrid>

                <Alert status="info" borderRadius="md">
                  <AlertIcon />
                  <Box>
                    <AlertTitle>Level Kepercayaan</AlertTitle>
                    <AlertDescription>
                      {(() => {
                        const explanation = recommendations.enhanced_confidence?.explanation;
                        if (explanation === 'Based on high activity and fat loss goal') return 'Berdasarkan aktivitas tinggi dan tujuan membakar lemak';
                        if (explanation === 'Based on moderate activity and muscle gain goal') return 'Berdasarkan aktivitas sedang dan tujuan menambah massa otot';
                        if (explanation === 'Based on low activity and maintenance goal') return 'Berdasarkan aktivitas rendah dan tujuan mempertahankan berat';
                        if (explanation === 'Based on moderate activity and maintenance goal') return 'Berdasarkan aktivitas sedang dan tujuan mempertahankan berat';
                        if (explanation === 'Based on high activity and muscle gain goal') return 'Berdasarkan aktivitas tinggi dan tujuan menambah massa otot';
                        if (explanation === 'Based on low activity and fat loss goal') return 'Berdasarkan aktivitas rendah dan tujuan membakar lemak';
                        return explanation || 'Rekomendasi dibuat berdasarkan data yang Anda berikan.';
                      })()}
                    </AlertDescription>
                  </Box>
                </Alert>
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Workout Recommendations */}
        {workout && (
          <Card>
            <CardHeader>
              <HStack justify="space-between">
                <Heading size={{ base: "sm", md: "md" }}>🏋️ Program Latihan Harian</Heading>
                {workout.template_id && (
                  <Badge colorScheme="blue" fontSize="0.8em">
                    Template ID: {workout.template_id}
                  </Badge>
                )}
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={{ base: 4, md: 6 }} align="stretch">
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={{ base: 4, md: 6 }}>
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={3}>📅 Jadwal Mingguan</Heading>
                    <SimpleGrid columns={2} spacing={3}>
                      <Stat>
                        <StatLabel>Jenis Olahraga</StatLabel>
                        <StatNumber>
                          {workout.workout_type === 'Full Body' ? 'Seluruh Tubuh' :
                           workout.workout_type === 'Upper/Lower Split' ? 'Split Atas/Bawah' :
                           workout.workout_type === 'Push/Pull/Legs' ? 'Dorong/Tarik/Kaki' :
                           workout.workout_type === 'Strength Training' ? 'Latihan Kekuatan' :
                           workout.workout_type || 'Latihan Kekuatan'}
                        </StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Hari per Minggu</StatLabel>
                        <StatNumber>{workout.days_per_week || 3} hari</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Set Harian</StatLabel>
                        <StatNumber>{workout.sets_per_exercise || 3} set</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Latihan per Sesi</StatLabel>
                        <StatNumber>{workout.exercises_per_session || 5} latihan</StatNumber>
                      </Stat>
                    </SimpleGrid>
                  </Box>

                  {workout.cardio_minutes_per_day && (
                    <Box>
                      <Heading size={{ base: "xs", md: "sm" }} mb={3}>🏃 Kardio Harian</Heading>
                      <SimpleGrid columns={2} spacing={3}>
                        <Stat>
                          <StatLabel>Durasi per Hari</StatLabel>
                          <StatNumber>{workout.cardio_minutes_per_day} menit</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Sesi per Hari</StatLabel>
                          <StatNumber>{workout.cardio_sessions_per_day || 1} sesi</StatNumber>
                        </Stat>
                      </SimpleGrid>
                    </Box>
                  )}
                </SimpleGrid>

                {(workout.workout_schedule || workout.schedule || workout.weekly_schedule) && (
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={3}>📋 Jadwal yang Disarankan</Heading>
                    <Box p={4} bg="gray.50" borderRadius="md">
                      <Text fontFamily="mono" fontSize={{ base: "md", md: "lg" }}>
                        {workout.workout_schedule || workout.schedule || workout.weekly_schedule}
                      </Text>
                      <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mt={2}>
                        W = Hari latihan, X = Hari istirahat
                      </Text>
                    </Box>
                  </Box>
                )}

                <Box>
                  <Heading size={{ base: "xs", md: "sm" }} mb={3}>📚 Penjelasan Jenis Latihan</Heading>
                  <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
                    <Box p={4} borderWidth={1} borderRadius="md">
                      <Text fontWeight="bold" mb={2} fontSize={{ base: "sm", md: "md" }}>🏋️ Full Body Workouts</Text>
                      <Text fontSize={{ base: "xs", md: "sm" }}>Melatih semua kelompok otot utama dalam setiap sesi. Cocok untuk pemula dan orang dengan waktu terbatas.</Text>
                    </Box>
                    <Box p={4} borderWidth={1} borderRadius="md">
                      <Text fontWeight="bold" mb={2} fontSize={{ base: "sm", md: "md" }}>🔄 Upper/Lower Split</Text>
                      <Text fontSize={{ base: "xs", md: "sm" }}>Split latihan atas dan bawah tubuh. Cocok untuk atlet menengah dengan ketersediaan waktu sedang.</Text>
                    </Box>
                    <Box p={4} borderWidth={1} borderRadius="md">
                      <Text fontWeight="bold" mb={2} fontSize={{ base: "sm", md: "md" }}>💪 Push/Pull/Legs</Text>
                      <Text fontSize={{ base: "xs", md: "sm" }}>Split berdasarkan gerakan dorong, tarik, dan kaki. Cocok untuk atlet lanjutan.</Text>
                    </Box>
                  </SimpleGrid>
                </Box>
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Nutrition Recommendations */}
        {nutrition && dailyMacros && (
          <Card>
            <CardHeader>
              <HStack justify="space-between">
                <Heading size={{ base: "sm", md: "md" }}>🍎 Program Nutrisi Harian</Heading>
                {nutrition.template_id && (
                  <Badge colorScheme="green" fontSize="0.8em">
                    Template ID: {nutrition.template_id}
                  </Badge>
                )}
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={{ base: 4, md: 6 }} align="stretch">
                <Box>
                  <Heading size={{ base: "xs", md: "sm" }} mb={4}>🎯 Target Harian Berdasarkan Template</Heading>
                  <SimpleGrid columns={{ base: 1, sm: 2, md: 4 }} spacing={{ base: 3, md: 4 }}>
                    <Stat>
                      <StatLabel>🔥 Kalori</StatLabel>
                      <StatNumber>{dailyMacros.calories} kkal</StatNumber>
                      <StatHelpText>
                        <Badge colorScheme={
                          userData?.fitness_goal === 'Fat Loss' ? 'red' : 
                          userData?.fitness_goal === 'Muscle Gain' ? 'green' : 'blue'
                        }>
                          {userData?.fitness_goal === 'Fat Loss' ? `Defisit: ${Math.round((1 - (dailyMacros.calories / (recommendations?.user_profile?.tdee || 2000))) * 100)}%` :
                           userData?.fitness_goal === 'Muscle Gain' ? `Surplus: +${Math.round(((dailyMacros.calories / (recommendations?.user_profile?.tdee || 2000)) - 1) * 100)}%` :
                           'Maintenance'}
                        </Badge>
                      </StatHelpText>
                    </Stat>
                    <Stat>
                      <StatLabel>🥩 Protein</StatLabel>
                      <StatNumber>{dailyMacros.protein}g</StatNumber>
                    </Stat>
                    <Stat>
                      <StatLabel>🍞 Karbohidrat</StatLabel>
                      <StatNumber>{dailyMacros.carbs}g</StatNumber>
                    </Stat>
                    <Stat>
                      <StatLabel>🥑 Lemak</StatLabel>
                      <StatNumber>{dailyMacros.fat}g</StatNumber>
                    </Stat>
                  </SimpleGrid>
                </Box>

                {/* Food Suggestions with Calculated Portions */}
                {(!mealPlanLoading && !backendMealPlanLoading) && foodSuggestions.length > 0 && (
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={4}>
                      🍽️ {mealPlan ? 'Rencana Makan Berdasarkan Template' : 'Porsi Makanan Indonesia Berdasarkan Template'}
                    </Heading>
                    <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mb={4}>
                      {mealPlan ? 'Kombinasi makanan yang sudah diatur untuk mencapai target nutrisi harian Anda' : 'Porsi yang dihitung untuk mencapai target nutrisi harian Anda'}
                    </Text>
                    
                    <VStack spacing={4} align="stretch">
                      {foodSuggestions.map((meal, index) => (
                        <Box key={index} p={4} borderWidth={1} borderRadius="md" bg="gray.50">
                          <Heading size={{ base: "xs", md: "sm" }} mb={2}>{meal.meal}</Heading>
                          {meal.mealName && (
                            <Box mb={2}>
                              <Text fontWeight="bold" fontSize={{ base: "xs", md: "sm" }}>{meal.mealName}</Text>
                              {meal.description && <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600">{meal.description}</Text>}
                            </Box>
                          )}
                          <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mb={3}>
                            Target: {meal.targetCalories} kkal | {meal.targetProtein}g protein | {meal.targetCarbs}g carbs | {meal.targetFat}g fat
                            {meal.actualCalories && (
                              <Text as="span" color="blue.600" ml={2}>
                                • Aktual: {meal.actualCalories} kkal | {meal.actualProtein}g protein | {meal.actualCarbs}g carbs | {meal.actualFat}g fat
                              </Text>
                            )}
                          </Text>
                          
                          <VStack spacing={2} align="stretch">
                            {meal.foods.map((food, foodIndex) => (
                              <Box key={foodIndex} p={3} bg="white" borderRadius="md" borderWidth={1}>
                                <Flex justify="space-between" align="center" mb={2}>
                                  <Text fontWeight="bold" fontSize={{ base: "xs", md: "sm" }}>{food.name}</Text>
                                  <Badge colorScheme="blue" fontSize={{ base: "2xs", md: "xs" }}>{food.grams}g</Badge>
                                </Flex>
                                <SimpleGrid columns={4} spacing={2}>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🔥 {food.actualCalories} kkal</Text>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🥩 {food.actualProtein}g</Text>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🍞 {food.actualCarbs}g</Text>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🥑 {food.actualFat}g</Text>
                                </SimpleGrid>
                              </Box>
                            ))}
                          </VStack>
                        </Box>
                      ))}
                    </VStack>
                    
                    {/* Total Summary */}
                    {foodSuggestions.length > 0 && foodSuggestions[0].actualCalories && (
                      <Box mt={4} p={4} bg="blue.50" borderRadius="md">
                        <Heading size={{ base: "xs", md: "sm" }} mb={3}>📊 Ringkasan Total Harian</Heading>
                        <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3}>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Total Kalori</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>
                              {foodSuggestions.reduce((sum, meal) => sum + (meal.actualCalories || 0), 0)} kkal
                            </StatNumber>
                            <StatHelpText fontSize={{ base: "2xs", md: "xs" }}>
                              Target: {dailyMacros.calories} kkal
                            </StatHelpText>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Total Protein</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>
                              {Math.round(foodSuggestions.reduce((sum, meal) => sum + (meal.actualProtein || 0), 0) * 10) / 10}g
                            </StatNumber>
                            <StatHelpText fontSize={{ base: "2xs", md: "xs" }}>
                              Target: {dailyMacros.protein}g
                            </StatHelpText>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Total Karbo</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>
                              {Math.round(foodSuggestions.reduce((sum, meal) => sum + (meal.actualCarbs || 0), 0) * 10) / 10}g
                            </StatNumber>
                            <StatHelpText fontSize={{ base: "2xs", md: "xs" }}>
                              Target: {dailyMacros.carbs}g
                            </StatHelpText>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Total Lemak</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>
                              {Math.round(foodSuggestions.reduce((sum, meal) => sum + (meal.actualFat || 0), 0) * 10) / 10}g
                            </StatNumber>
                            <StatHelpText fontSize={{ base: "2xs", md: "xs" }}>
                              Target: {dailyMacros.fat}g
                            </StatHelpText>
                          </Stat>
                        </SimpleGrid>
                        
                        {/* Accuracy indicators */}
                        <Box mt={4}>
                          <Text fontSize={{ base: "2xs", md: "xs" }} fontWeight="bold" mb={2}>📈 Akurasi Target:</Text>
                          <SimpleGrid columns={{ base: 2, md: 4 }} spacing={2}>
                            {(() => {
                              const totalActualCalories = foodSuggestions.reduce((sum, meal) => sum + (meal.actualCalories || 0), 0);
                              const totalActualProtein = foodSuggestions.reduce((sum, meal) => sum + (meal.actualProtein || 0), 0);
                              const totalActualCarbs = foodSuggestions.reduce((sum, meal) => sum + (meal.actualCarbs || 0), 0);
                              const totalActualFat = foodSuggestions.reduce((sum, meal) => sum + (meal.actualFat || 0), 0);
                              
                              const calorieAccuracy = Math.round((totalActualCalories / dailyMacros.calories) * 100);
                              const proteinAccuracy = Math.round((totalActualProtein / dailyMacros.protein) * 100);
                              const carbAccuracy = Math.round((totalActualCarbs / dailyMacros.carbs) * 100);
                              const fatAccuracy = Math.round((totalActualFat / dailyMacros.fat) * 100);
                              
                              return (
                                <>
                                  <Badge colorScheme={calorieAccuracy >= 90 && calorieAccuracy <= 110 ? 'green' : calorieAccuracy >= 80 && calorieAccuracy <= 120 ? 'yellow' : 'red'} fontSize={{ base: "2xs", md: "xs" }}>
                                    Kalori: {calorieAccuracy}%
                                  </Badge>
                                  <Badge colorScheme={proteinAccuracy >= 90 && proteinAccuracy <= 110 ? 'green' : proteinAccuracy >= 80 && proteinAccuracy <= 120 ? 'yellow' : 'red'} fontSize={{ base: "2xs", md: "xs" }}>
                                    Protein: {proteinAccuracy}%
                                  </Badge>
                                  <Badge colorScheme={carbAccuracy >= 90 && carbAccuracy <= 110 ? 'green' : carbAccuracy >= 80 && carbAccuracy <= 120 ? 'yellow' : 'red'} fontSize={{ base: "2xs", md: "xs" }}>
                                    Karbo: {carbAccuracy}%
                                  </Badge>
                                  <Badge colorScheme={fatAccuracy >= 90 && fatAccuracy <= 110 ? 'green' : fatAccuracy >= 80 && fatAccuracy <= 120 ? 'yellow' : 'red'} fontSize={{ base: "2xs", md: "xs" }}>
                                    Lemak: {fatAccuracy}%
                                  </Badge>
                                </>
                              );
                            })()}
                          </SimpleGrid>
                        </Box>
                      </Box>
                    )}
                  </Box>
                )}
                
                {/* Backend AI-Generated Meal Plan */}
                {!backendMealPlanLoading && backendMealPlan && (
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={4}>🤖 AI-Generated Meal Plan</Heading>
                    <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mb={4}>
                      Rencana makan yang dihasilkan oleh AI berdasarkan template nutrisi dan kebutuhan kalori Anda
                    </Text>
                    
                    {/* Daily Summary */}
                    {backendMealPlan.daily_summary && (
                      <Box mb={4} p={4} bg="blue.50" borderRadius="md">
                        <Heading size={{ base: "xs", md: "sm" }} mb={3}>📊 Ringkasan Harian</Heading>
                        <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3}>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Total Kalori</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_calories)} kkal</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Protein</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_protein)}g</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Karbohidrat</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_carbs)}g</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Lemak</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_fat)}g</StatNumber>
                          </Stat>
                        </SimpleGrid>
                      </Box>
                    )}

                    {/* Detailed Meals */}
                    {backendMealPlan.meals && (
                      <VStack spacing={4} align="stretch">
                        {Object.entries(backendMealPlan.meals).map(([mealType, mealData]) => (
                          <Box key={mealType} p={4} borderWidth={1} borderRadius="md" bg="green.50">
                            <Heading size={{ base: "xs", md: "sm" }} mb={3}>
                              {mealType === 'breakfast' && '🌅 Sarapan'}
                              {mealType === 'morning_snack' && '🍎 Snack Pagi'}
                              {mealType === 'lunch' && '🌞 Makan Siang'}
                              {mealType === 'afternoon_snack' && '🥜 Snack Sore'}
                              {mealType === 'dinner' && '🌙 Makan Malam'}
                              {mealType === 'evening_snack' && '🍪 Snack Malam'}
                            </Heading>
                            
                            <SimpleGrid columns={4} spacing={2} mb={3}>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🔥 {Math.round(mealData.calories)} kkal</Text>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🥩 {Math.round(mealData.protein)}g</Text>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🍞 {Math.round(mealData.carbs)}g</Text>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🥑 {Math.round(mealData.fat)}g</Text>
                            </SimpleGrid>

                            <VStack spacing={2} align="stretch">
                              {mealData.foods && mealData.foods.map((food, foodIndex) => (
                                <Box key={foodIndex} p={3} bg="white" borderRadius="md" borderWidth={1}>
                                  <Flex justify="space-between" align="center" mb={2}>
                                    <Text fontWeight="bold" fontSize={{ base: "xs", md: "sm" }}>{food.name}</Text>
                                    <Badge colorScheme="green" fontSize={{ base: "2xs", md: "xs" }}>{food.portion}g</Badge>
                                  </Flex>
                                  <SimpleGrid columns={4} spacing={2}>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🔥 {Math.round(food.calories)} kkal</Text>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🥩 {Math.round(food.protein)}g</Text>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🍞 {Math.round(food.carbs)}g</Text>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🥑 {Math.round(food.fat)}g</Text>
                                  </SimpleGrid>
                                </Box>
                              ))}
                            </VStack>
                          </Box>
                        ))}
                      </VStack>
                    )}

                    {/* Shopping List */}
                    {backendMealPlan.shopping_list && backendMealPlan.shopping_list.length > 0 && (
                      <Box mt={4} p={4} bg="orange.50" borderRadius="md">
                        <Heading size={{ base: "xs", md: "sm" }} mb={3}>🛒 Daftar Belanja</Heading>
                        <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={2}>
                          {backendMealPlan.shopping_list.map((item, index) => (
                            <Flex key={index} justify="space-between" p={2} bg="white" borderRadius="md">
                              <Text fontSize={{ base: "xs", md: "sm" }}>{item.name}</Text>
                              <Text fontSize={{ base: "xs", md: "sm" }} fontWeight="bold">{item.total_amount}g</Text>
                            </Flex>
                          ))}
                        </SimpleGrid>
                      </Box>
                    )}
                  </Box>
                )}
                
                {/* Loading state for backend meal plan */}
                {backendMealPlanLoading && (
                  <Box p={6} textAlign="center" bg="blue.50" borderRadius="md">
                    <Heading size={{ base: "sm", md: "md" }} mb={2}>🤖 Memuat AI Meal Plan...</Heading>
                    <Text fontSize={{ base: "sm", md: "md" }}>Sedang menghasilkan rencana makan yang dipersonalisasi dengan AI...</Text>
                  </Box>
                )}
                
                {/* Loading state for meal plan */}
                {mealPlanLoading && (
                  <Box p={6} textAlign="center" bg="green.50" borderRadius="md">
                    <Heading size={{ base: "sm", md: "md" }} mb={2}>🔄 Memuat Rencana Makan...</Heading>
                    <Text fontSize={{ base: "sm", md: "md" }}>Sedang menyusun kombinasi makanan yang optimal untuk Anda...</Text>
                  </Box>
                )}
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Action Buttons */}
        <VStack spacing={4}>
          <HStack justify="center" spacing={{ base: 2, md: 4 }} w="full">
            <Button 
              variant="outline" 
              colorScheme="brand" 
              onClick={onBack}
              size={{ base: "sm", md: "md" }}
              w={{ base: "full", md: "auto" }}
            >
              ← Kembali ke Formulir
            </Button>
            <Button 
              colorScheme="brand" 
              onClick={onNewRecommendation}
              size={{ base: "sm", md: "md" }}
              w={{ base: "full", md: "auto" }}
            >
              🆕 Buat Rekomendasi Baru
            </Button>
          </HStack>
        </VStack>

        {/* Tips Section */}
        <Card>
          <CardHeader>
            <Heading size={{ base: "sm", md: "md" }}>💡 Tips Sukses dengan Porsi yang Tepat</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, sm: 2, lg: 4 }} spacing={{ base: 3, md: 4 }}>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>⚖️</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Gunakan timbangan digital untuk mengukur porsi makanan secara akurat</Text>
              </Box>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>📱</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Catat asupan makanan harian di fitur progress tracking</Text>
              </Box>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>🥗</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Variasikan sumber protein dan karbohidrat setiap harinya</Text>
              </Box>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>💧</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Minum 2-3 liter air putih setiap hari untuk metabolisme optimal</Text>
              </Box>
            </SimpleGrid>
          </CardBody>
        </Card>
      </VStack>
    </Box>
  );
};

export default RecommendationDisplay;