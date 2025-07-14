import React, { useState } from "react";
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Input,
  FormControl,
  FormLabel,
  FormHelperText,
  InputGroup,
  InputRightElement,
  IconButton,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useToast,
  useColorModeValue,
  Skeleton,
  SkeletonText,
  List,
  ListItem,
  ListIcon,
  Badge,
  Flex,
  Spacer,
  useDisclosure,
} from "@chakra-ui/react";
import {
  ViewIcon,
  ViewOffIcon,
  EmailIcon,
  LockIcon,
  CheckIcon,
  InfoIcon,
} from "@chakra-ui/icons";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  updateProfile,
} from "firebase/auth";
import { auth } from "../services/firebaseConfig";

const AuthForm = ({ onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    displayName: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const toast = useToast();

  // Color mode values
  const bgColor = useColorModeValue("white", "gray.800");
  const borderColor = useColorModeValue("gray.200", "gray.700");
  const cardBg = useColorModeValue("white", "gray.700");
  const textColor = useColorModeValue("gray.800", "white");
  const mutedTextColor = useColorModeValue("gray.600", "gray.400");
  const gradientBg = useColorModeValue(
    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)"
  );

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      let userCredential;
      if (isLogin) {
        userCredential = await signInWithEmailAndPassword(
          auth,
          formData.email,
          formData.password
        );
      } else {
        userCredential = await createUserWithEmailAndPassword(
          auth,
          formData.email,
          formData.password
        );
        if (formData.displayName) {
          await updateProfile(userCredential.user, {
            displayName: formData.displayName,
          });
        }
      }

      // Call success callback with user data
      if (onAuthSuccess && userCredential.user) {
        onAuthSuccess(userCredential.user);
      }
    } catch (error) {
      let errorMessage = "Terjadi kesalahan, silakan coba lagi.";

      switch (error.code) {
        case "auth/user-not-found":
          errorMessage = "Email tidak ditemukan.";
          break;
        case "auth/wrong-password":
          errorMessage = "Password salah.";
          break;
        case "auth/email-already-in-use":
          errorMessage = "Email sudah terdaftar.";
          break;
        case "auth/weak-password":
          errorMessage = "Password terlalu lemah. Minimal 6 karakter.";
          break;
        case "auth/invalid-email":
          errorMessage = "Format email tidak valid.";
          break;
        case "auth/too-many-requests":
          errorMessage = "Terlalu banyak percobaan. Silakan coba lagi nanti.";
          break;
        default:
          errorMessage = error.message;
      }

      setError(errorMessage);
      toast({
        title: "Error",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError("");

    try {
      const provider = new GoogleAuthProvider();
      const userCredential = await signInWithPopup(auth, provider);

      // Call success callback with user data
      if (onAuthSuccess && userCredential.user) {
        onAuthSuccess(userCredential.user);
      }
    } catch (error) {
      let errorMessage = "Gagal masuk dengan Google.";

      if (error.code === "auth/popup-closed-by-user") {
        errorMessage = "Popup ditutup. Silakan coba lagi.";
      } else if (error.code === "auth/network-request-failed") {
        errorMessage = "Periksa koneksi internet Anda.";
      }

      setError(errorMessage);
      toast({
        title: "Error",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      minH="100vh"
      bg={gradientBg}
      display="flex"
      alignItems="center"
      justifyContent="center"
      p={{ base: 4, md: 6 }}
    >
      <Box maxW={{ base: "full", sm: "md", md: "lg" }} w="full" mx="auto">
        <Card
          bg={cardBg}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="xl"
          boxShadow="xl"
          overflow="hidden"
        >
          {/* Header */}
          <Box
            bg={gradientBg}
            p={{ base: 6, md: 8 }}
            textAlign="center"
            color="white"
          >
            <VStack spacing={3}>
              <Box
                w="60px"
                h="60px"
                bg="white"
                borderRadius="full"
                display="flex"
                alignItems="center"
                justifyContent="center"
                boxShadow="lg"
              >
                <Text
                  fontSize="2xl"
                  fontWeight="bold"
                  bg={gradientBg}
                  bgClip="text"
                >
                  💪
                </Text>
              </Box>
              <Heading size={{ base: "lg", md: "xl" }}>
                {isLogin
                  ? "Selamat Datang Kembali"
                  : "Bergabung dengan XGFitness"}
              </Heading>
              <Text fontSize={{ base: "sm", md: "md" }} opacity={0.9}>
                {isLogin
                  ? "Masuk untuk melanjutkan perjalanan fitness Anda"
                  : "Mulai perjalanan fitness yang dipersonalisasi"}
              </Text>
            </VStack>
          </Box>

          <CardBody p={{ base: 6, md: 8 }}>
            <VStack spacing={6} align="stretch">
              {/* Error Alert */}
              {error && (
                <Alert status="error" borderRadius="md">
                  <AlertIcon />
                  <Box flex={1}>
                    <AlertTitle>Oops! Ada masalah</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Box>
                </Alert>
              )}

              {/* Auth Form */}
              <form onSubmit={handleSubmit}>
                <VStack spacing={4} align="stretch">
                  {/* Display Name (Register only) */}
                  {!isLogin && (
                    <FormControl isRequired>
                      <FormLabel fontSize={{ base: "sm", md: "md" }}>
                        <HStack spacing={2}>
                          <InfoIcon color="blue.500" />
                          <Text>Nama Lengkap</Text>
                        </HStack>
                      </FormLabel>
                      <Input
                        type="text"
                        name="displayName"
                        value={formData.displayName}
                        onChange={handleInputChange}
                        placeholder="Masukkan nama lengkap Anda"
                        size={{ base: "md", md: "lg" }}
                        bg={bgColor}
                        borderColor={borderColor}
                        _focus={{
                          borderColor: "blue.500",
                          boxShadow: "0 0 0 1px var(--chakra-colors-blue-500)",
                        }}
                      />
                    </FormControl>
                  )}

                  {/* Email */}
                  <FormControl isRequired>
                    <FormLabel fontSize={{ base: "sm", md: "md" }}>
                      <HStack spacing={2}>
                        <EmailIcon color="blue.500" />
                        <Text>Email</Text>
                      </HStack>
                    </FormLabel>
                    <Input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      placeholder="nama@email.com"
                      size={{ base: "md", md: "lg" }}
                      bg={bgColor}
                      borderColor={borderColor}
                      _focus={{
                        borderColor: "blue.500",
                        boxShadow: "0 0 0 1px var(--chakra-colors-blue-500)",
                      }}
                    />
                  </FormControl>

                  {/* Password */}
                  <FormControl isRequired>
                    <FormLabel fontSize={{ base: "sm", md: "md" }}>
                      <HStack spacing={2}>
                        <LockIcon color="blue.500" />
                        <Text>Password</Text>
                      </HStack>
                    </FormLabel>
                    <InputGroup size={{ base: "md", md: "lg" }}>
                      <Input
                        type={showPassword ? "text" : "password"}
                        name="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        placeholder="Masukkan password"
                        bg={bgColor}
                        borderColor={borderColor}
                        _focus={{
                          borderColor: "blue.500",
                          boxShadow: "0 0 0 1px var(--chakra-colors-blue-500)",
                        }}
                        minLength={6}
                      />
                      <InputRightElement>
                        <IconButton
                          icon={showPassword ? <ViewOffIcon /> : <ViewIcon />}
                          onClick={() => setShowPassword(!showPassword)}
                          variant="ghost"
                          size="sm"
                          aria-label={
                            showPassword ? "Hide password" : "Show password"
                          }
                        />
                      </InputRightElement>
                    </InputGroup>
                    {!isLogin && (
                      <FormHelperText fontSize="xs" color={mutedTextColor}>
                        Password minimal 6 karakter
                      </FormHelperText>
                    )}
                  </FormControl>

                  {/* Submit Button */}
                  <Button
                    type="submit"
                    colorScheme="blue"
                    size={{ base: "md", md: "lg" }}
                    isLoading={loading}
                    loadingText={isLogin ? "Masuk..." : "Mendaftar..."}
                    w="full"
                    fontWeight="bold"
                    py={3}
                  >
                    {isLogin ? "Masuk" : "Daftar"}
                  </Button>
                </VStack>
              </form>

              {/* Switch Mode */}
              <Box textAlign="center">
                <Text
                  fontSize={{ base: "sm", md: "md" }}
                  color={mutedTextColor}
                >
                  {isLogin ? "Belum punya akun?" : "Sudah punya akun?"}
                  <Button
                    variant="link"
                    colorScheme="blue"
                    ml={2}
                    onClick={() => {
                      setIsLogin(!isLogin);
                      setError("");
                      setFormData({ email: "", password: "", displayName: "" });
                    }}
                    fontSize={{ base: "sm", md: "md" }}
                  >
                    {isLogin ? "Daftar di sini" : "Masuk di sini"}
                  </Button>
                </Text>
              </Box>
            </VStack>
          </CardBody>
        </Card>

        {/* Features Section (Register only) */}
        {!isLogin && (
          <Card
            mt={6}
            bg={cardBg}
            borderWidth="1px"
            borderColor={borderColor}
            borderRadius="xl"
          >
            <CardHeader>
              <Heading size={{ base: "md", md: "lg" }} textAlign="center">
                Mengapa bergabung dengan XGFitness?
              </Heading>
            </CardHeader>
            <CardBody>
              <List spacing={3}>
                {[
                  {
                    icon: "🎯",
                    text: "Rekomendasi workout yang dipersonalisasi",
                  },
                  {
                    icon: "🍎",
                    text: "Panduan nutrisi lengkap dengan makanan Indonesia",
                  },
                  {
                    icon: "📊",
                    text: "Tracking progress yang mudah dan akurat",
                  },
                  {
                    icon: "🤖",
                    text: "AI yang terus belajar dari kebiasaan Anda",
                  },
                  { icon: "🏆", text: "Pencapaian dan motivasi harian" },
                ].map((feature, index) => (
                  <ListItem key={index}>
                    <HStack spacing={3}>
                      <Text fontSize="lg">{feature.icon}</Text>
                      <Text fontSize={{ base: "sm", md: "md" }}>
                        {feature.text}
                      </Text>
                    </HStack>
                  </ListItem>
                ))}
              </List>
            </CardBody>
          </Card>
        )}
      </Box>
    </Box>
  );
};

export default AuthForm;
